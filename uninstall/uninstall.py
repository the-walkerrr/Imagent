"""
uninstall/uninstall.py — Take Imagent off this computer: show what will be lost, ask three times, then delete.

Started by uninstall/uninstall.sh / uninstall/uninstall.bat with a Python that is not the one in .venv, because it deletes .venv.
Standard library only. It removes:
  * the Ollama models that install.py downloaded (listed in install_record.json); a model you already had
    before Imagent is never removed;
  * Imagent's data folder (the index database, config.json, logs);
  * the .venv folder and every __pycache__ folder in the project.
It never touches your image files, this project's source files, Ollama itself or any other Ollama model.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

# Make the project root importable when this file is run directly (``python uninstall/uninstall.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.console import make_output_safe, run_cli
from pipeline.locking import AlreadyRunning, single_instance

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_DIR / ".venv"
DEFAULT_HOST = "http://localhost:11434"
FINAL_PHRASE = "uninstall imagent"
CANCELLED = "Uninstall cancelled. Nothing was deleted."


# --- Sizes and numbers --------------------------------------------------------

def _folder_size(path: Path) -> int:
    """Bytes in the files under ``path`` (links are not followed); 0 if it does not exist."""
    total = 0
    for folder, _, files in os.walk(path):
        for name in files:
            try:
                status = os.lstat(os.path.join(folder, name))
            except OSError:
                continue
            if stat.S_ISREG(status.st_mode):
                total += status.st_size
    return total


def _fmt_bytes(n: float) -> str:
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{int(n)} B"


def _fmt_duration(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} s"


# --- What is in the database ---------------------------------------------------

def read_stats() -> dict:
    """Counts from the index database, read-only. Every value is None when it cannot be read (no database, damaged)."""
    stats = {"images": None, "folders": [], "queued": None, "errors": None, "analysis_ms": None, "locked_models": []}
    if not config.DB_PATH.exists():
        return stats
    try:
        conn = sqlite3.connect(f"{config.DB_PATH.as_uri()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return stats
    try:
        for key, sql in (("images", "SELECT COUNT(*) FROM image_data"), ("queued", "SELECT COUNT(*) FROM backlog"),
                         ("errors", "SELECT COUNT(*) FROM error_logs"),
                         ("analysis_ms", "SELECT COALESCE(SUM(processing_time_ms), 0) FROM image_data")):
            try:
                stats[key] = conn.execute(sql).fetchone()[0]
            except sqlite3.Error:
                pass
        try:
            stats["folders"] = [row[0] for row in conn.execute("SELECT path FROM scan_roots ORDER BY path")]
        except sqlite3.Error:
            pass
        try:
            row = conn.execute("SELECT vlm_model, embedding_model, judge_model FROM model_settings").fetchone()
            stats["locked_models"] = list(dict.fromkeys(row)) if row else []
        except sqlite3.Error:
            pass
    finally:
        conn.close()
    return stats


# --- Ollama (plain HTTP, so no package is needed) --------------------------------

def _base_url(host: str) -> str:
    """Same rule as vlm.client: a host with no scheme gets Ollama's port 11434, one with a scheme the scheme's port."""
    has_scheme = "://" in host
    parsed = urlparse(host if has_scheme else f"http://{host}")
    port = parsed.port or (11434 if not has_scheme else 443 if parsed.scheme == "https" else 80)
    name = parsed.hostname or "127.0.0.1"
    return f"{parsed.scheme}://{f'[{name}]' if ':' in name else name}:{port}"


def _is_local(host: str) -> bool:
    parsed = urlparse(host if "://" in host else f"http://{host}")
    return parsed.hostname in ("localhost", "127.0.0.1", "::1")


def _call(base: str, method: str, path: str, body: dict | None = None, timeout: int = 10):
    request = urllib.request.Request(
        f"{base}{path}", method=method, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


def _reachable(base: str) -> bool:
    try:
        _call(base, "GET", "/api/version", timeout=3)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


@contextmanager
def _ollama(host: str):
    """Yield the server's base URL once Ollama answers, or None. Starts ``ollama serve`` only if needed, and stops it afterwards."""
    base = _base_url(host)
    process = None
    try:
        if not _reachable(base):
            if _is_local(host) and shutil.which("ollama"):
                kwargs = {"creationflags": 0x00000008} if platform.system() == "Windows" else {"start_new_session": True}
                try:
                    process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               env={**os.environ, "OLLAMA_HOST": host}, **kwargs)
                except OSError:
                    process = None
                for _ in range(30 if process else 0):
                    if _reachable(base):
                        break
                    time.sleep(1)
        yield base if _reachable(base) else None
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def _installed_models(base: str | None) -> dict[str, int] | None:
    """{model name: size in bytes} of what Ollama has, or None if it cannot be asked."""
    if base is None:
        return None
    try:
        return {m["name"]: int(m.get("size") or 0) for m in _call(base, "GET", "/api/tags")["models"]}
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return None


# --- The plan -------------------------------------------------------------------

def make_plan(base: str | None, stats: dict) -> dict:
    """Work out what will be removed, and what is left alone and why."""
    recorded = [m for m in config.load_install_record().get("ollama_models", []) if isinstance(m, str)]
    present = _installed_models(base)
    plan = {
        "models": [], "models_unreachable": [], "models_gone": [],
        "not_recorded": [m for m in stats["locked_models"] if m not in recorded],
        "data_dir": config.IMAGENT_HOME if config.IMAGENT_HOME.exists() else None,
        "venv": VENV_DIR if VENV_DIR.exists() else None,
    }
    for name in recorded:
        if present is None:
            plan["models_unreachable"].append(name)
        elif name in present:
            plan["models"].append((name, present[name]))
        else:
            plan["models_gone"].append(name)
    plan["data_size"] = _folder_size(plan["data_dir"]) if plan["data_dir"] else 0
    plan["venv_size"] = _folder_size(plan["venv"]) if plan["venv"] else 0
    plan["db_size"] = sum(config.DB_PATH.with_name(config.DB_PATH.name + suffix).stat().st_size
                          for suffix in ("", "-wal", "-shm")
                          if config.DB_PATH.with_name(config.DB_PATH.name + suffix).exists())
    plan["models_size"] = sum(size for _, size in plan["models"])
    return plan


def nothing_to_remove(plan: dict) -> bool:
    return not (plan["data_dir"] or plan["venv"] or plan["models"])


def show_plan(stats: dict, plan: dict) -> None:
    def row(label: str, value: str, detail: str = "") -> None:
        print(f"  {label:<24}{value}" + (f"   {detail}" if detail else ""))

    unknown = "unknown"
    print("\n" + "=" * 64)
    print("  UNINSTALLING IMAGENT DELETES ALL OF THIS, AND IT CANNOT BE UNDONE")
    print("=" * 64)
    row("Indexed images", str(stats["images"]) if stats["images"] is not None else unknown,
        "(descriptions, tags, text and search vectors)")
    folders = stats["folders"]
    row("Scanned folders", str(len(folders)) if folders or stats["images"] is not None else unknown)
    for folder in folders:
        print(f"{'':<26}{folder}")
    row("Waiting to be indexed", str(stats["queued"]) if stats["queued"] is not None else unknown)
    row("Logged errors", str(stats["errors"]) if stats["errors"] is not None else unknown)
    if stats["analysis_ms"]:
        row("Time spent analysing", _fmt_duration(stats["analysis_ms"]), "(you would have to run it all again)")
    print("  " + "-" * 62)
    row("Index database", _fmt_bytes(plan["db_size"]), str(config.DB_PATH))
    row("Imagent data folder", _fmt_bytes(plan["data_size"]), str(plan["data_dir"] or "(none)"))
    row("Python environment", _fmt_bytes(plan["venv_size"]), f"{plan['venv'] or '.venv (none)'}  packages and text-reading models")
    row("Ollama models", _fmt_bytes(plan["models_size"]), "downloaded by Imagent" if plan["models"] else "(none to remove)")
    for name, size in plan["models"]:
        print(f"{'':<26}{name}  ({_fmt_bytes(size)})")
    total = plan["data_size"] + plan["venv_size"] + plan["models_size"]
    print("  " + "-" * 62)
    row("Disk space freed", _fmt_bytes(total))
    print("=" * 64)
    print("  Not touched: your image files, this project's source files, Ollama itself and")
    print("  every Ollama model that Imagent did not download.")
    if plan["models_unreachable"]:
        print("\n  Could not reach Ollama, so these models will NOT be removed (remove them yourself with")
        print("  `ollama rm <name>` on the computer where Ollama runs): " + ", ".join(plan["models_unreachable"]))
    if plan["not_recorded"]:
        print("\n  Not removed, because Imagent has no record of downloading them (you may have had them, or")
        print("  they were downloaded before Imagent kept a record): " + ", ".join(plan["not_recorded"]))
        print("  To remove one yourself: ollama rm <name>")


# --- Three confirmations ------------------------------------------------------------

def confirm_three_times(stats: dict) -> bool:
    """Ask three different questions; any wrong answer cancels everything."""
    if input("\n  Confirmation 1 of 3. Do you want to uninstall Imagent and delete everything above? Type yes: "
             ).strip().lower() != "yes":
        return False
    expected = str(stats["images"] if stats["images"] is not None else 0)
    if input("  Confirmation 2 of 3. How many indexed images will be lost? (see the list above): ").strip() != expected:
        print("  That is not the number in the list.")
        return False
    if input(f"  Confirmation 3 of 3. This is the last chance. Type exactly \"{FINAL_PHRASE}\": ").strip() != FINAL_PHRASE:
        return False
    return True


# --- Deleting -----------------------------------------------------------------------

def _remove_tree(path: Path) -> bool:
    """Delete a folder, clearing read-only flags (Windows) and retrying. Returns whether it is gone."""
    def clear_and_retry(function, target, _):
        try:
            os.chmod(target, stat.S_IWRITE)
            function(target)
        except OSError:
            pass
    option = {"onexc": clear_and_retry} if sys.version_info >= (3, 12) else {"onerror": clear_and_retry}
    shutil.rmtree(path, **option)
    return not path.exists()


def delete_everything(plan: dict, base: str | None) -> list[str]:
    """Do the deleting, one step at a time, and return the problems (empty when everything went well)."""
    problems = []
    for name, _ in plan["models"]:
        try:
            _call(base, "DELETE", "/api/delete", {"model": name}, timeout=120)
            print(f"  ✓ Removed the Ollama model {name}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            problems.append(f"Could not remove the Ollama model {name} ({exc}). Remove it with: ollama rm {name}")
    if plan["data_dir"]:
        if config.IMAGENT_HOME.name.lower() != "imagent":  # never delete a folder this program did not name itself
            problems.append(f"Did not delete {config.IMAGENT_HOME}: it is not named like Imagent's own folder.")
        elif _remove_tree(config.IMAGENT_HOME):
            print(f"  ✓ Deleted the data folder {config.IMAGENT_HOME}")
        else:
            problems.append(f"Could not delete everything in {config.IMAGENT_HOME}. Delete the folder yourself.")
    if plan["venv"]:
        if _remove_tree(VENV_DIR):
            print(f"  ✓ Deleted {VENV_DIR}")
        else:
            problems.append(f"Could not delete everything in {VENV_DIR}. Delete the folder yourself.")
    for cache in PROJECT_DIR.rglob("__pycache__"):
        _remove_tree(cache)
    return problems


# --- Entry point -------------------------------------------------------------------------

def main() -> None:
    make_output_safe()
    if not sys.stdin.isatty():
        sys.exit("[imagent] Uninstalling needs an interactive terminal. Run: sh uninstall/uninstall.sh (or uninstall/uninstall.bat on Windows)")
    if VENV_DIR.exists() and Path(sys.prefix).resolve() == VENV_DIR.resolve():
        sys.exit("[imagent] This is running inside Imagent's own .venv, which it needs to delete. "
                 "Run `deactivate` (or open a new terminal) and start the uninstaller again.")

    stats = read_stats()
    host = config.load_config().get("ollama_host") or DEFAULT_HOST
    print("\n  Looking at what Imagent has added…")
    with _ollama(host) as base:
        plan = make_plan(base, stats)
        if nothing_to_remove(plan):
            print("\n  Nothing to uninstall: Imagent has not added anything to this computer (or it was already removed).")
            return
        show_plan(stats, plan)
        if not confirm_three_times(stats):
            sys.exit(f"\n[imagent] {CANCELLED}")
        if config.IMAGENT_HOME.exists():
            try:
                with single_instance("indexing", "Indexing"):  # deleting under a running index would make it fail on every image
                    pass
            except AlreadyRunning as exc:
                sys.exit(f"\n[imagent] {exc} Nothing was deleted.")
        print()
        problems = delete_everything(plan, base)

    print()
    if problems:
        print("  Finished, with problems:")
        for problem in problems:
            print(f"  ✗ {problem}")
    else:
        print("  ✓ Imagent is uninstalled.")
    print(f"  To finish, delete this project folder if you no longer want it: {PROJECT_DIR}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    run_cli(main, cancelled=CANCELLED)
