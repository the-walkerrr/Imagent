"""
imagent/commands/uninstall.py — Take Imagent off this computer: show what will be lost, ask three times, then delete.

Run by ``imagent uninstall``. It removes:
  * the Ollama models that setup downloaded or copied (listed in install_record.json); a model you already had
    before Imagent is never removed;
  * Imagent's data folder (the index database, config.json, logs);
and, last, each program setup installed for Imagent (its own copy of Ollama, when the computer had none): for each one it describes
the program and asks whether to remove it too, and the answer is no unless you type y.
It never touches your image files, any other Ollama model, or an Ollama you had before Imagent. The program itself is removed afterwards
with ``pipx uninstall imagent`` (a running program cannot remove itself), and the last message says so.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sqlite3
import stat
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from imagent import config
from imagent.pipeline.console import make_output_safe
from imagent.os_specific.process import background_process_options
from imagent.pipeline.locking import AlreadyRunning, single_instance

DEFAULT_HOST = "http://localhost:11434"
FINAL_CODE_LENGTH = 10
CANCELLED = "Uninstall cancelled. Nothing was deleted."
REMOVE_PROGRAM = "To finish, remove the program itself with: pipx uninstall imagent   (or: pip uninstall imagent, if you used pip)"


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
    days, rest = divmod(milliseconds // 1000, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    if days:
        return f"{days} d {hours} h {minutes} min {seconds} s"
    if hours:
        return f"{hours} h {minutes} min {seconds} s"
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} s"


def _style(text: str, code: str) -> str:
    """``text`` with the ANSI style ``code`` when printing to a terminal that understands it, otherwise unchanged."""
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _style(text, "1")


def _green(text: str) -> str:
    return _style(text, "32")


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
            if _is_local(host) and config.ollama_binary():
                try:
                    process = subprocess.Popen([config.ollama_binary(), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               env=config.ollama_environment(host),
                                               **background_process_options())
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
        "tools": config.installed_tools(),
    }
    for name in recorded:
        if present is None:
            plan["models_unreachable"].append(name)
        elif name in present:
            plan["models"].append((name, present[name]))
        else:
            plan["models_gone"].append(name)
    plan["data_size"] = _folder_size(plan["data_dir"]) if plan["data_dir"] else 0
    plan["db_size"] = sum(config.DB_PATH.with_name(config.DB_PATH.name + suffix).stat().st_size
                          for suffix in ("", "-wal", "-shm")
                          if config.DB_PATH.with_name(config.DB_PATH.name + suffix).exists())
    plan["models_size"] = sum(size for _, size in plan["models"])
    return plan


def nothing_to_remove(plan: dict) -> bool:
    return not (plan["data_dir"] or plan["models"])


def show_plan(stats: dict, plan: dict) -> None:
    unknown = "unknown"
    width = 22  # the label column; values and folder paths start right after it

    def row(label: str, value: str) -> None:
        print(f"  {_bold(label)}{' ' * (width - len(label))}{value}")

    print("\n  Uninstalling Imagent deletes all of this, and it cannot be undone.")
    print(f"  {_green('Your images are not affected and stay the same.')}\n")
    row("Indexed images", str(stats["images"]) if stats["images"] is not None else unknown)
    folders = stats["folders"]
    row("Scanned folders", str(len(folders)) if folders or stats["images"] is not None else unknown)
    for folder in folders:
        print(f"  {' ' * width}{folder}")
    row("Total time analyzed", _fmt_duration(stats["analysis_ms"]) if stats["analysis_ms"] is not None else unknown)
    row("Storage used", _fmt_bytes(plan["data_size"] + plan["models_size"]))
    if plan["models_unreachable"]:
        print("\n  Could not reach Ollama, so these models will NOT be removed (remove them yourself with")
        print("  `ollama rm <name>` on the computer where Ollama runs): " + ", ".join(plan["models_unreachable"]))
    if plan["not_recorded"]:
        print("\n  Not removed, because Imagent has no record of downloading them (you may have had them, or")
        print("  they were downloaded before Imagent kept a record): " + ", ".join(plan["not_recorded"]))
        print("  To remove one yourself: ollama rm <name>")


# --- Three confirmations ------------------------------------------------------------

def make_final_code() -> str:
    """A new random string of letters and digits, typed in the last confirmation so it cannot be answered from habit."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(FINAL_CODE_LENGTH))


def confirm_three_times(stats: dict) -> bool:
    """Ask three different questions; any wrong answer cancels everything."""
    print(f"\n  {_bold('Confirmation')}")
    if input("  1/3  Do you want to uninstall Imagent and delete everything above? Type yes: ").strip().lower() != "yes":
        return False
    expected = str(stats["images"] if stats["images"] is not None else 0)
    if input("  2/3  How many indexed images will be lost? (see the list above): ").strip() != expected:
        print("  That is not the number in the list.")
        return False
    code = make_final_code()
    if input(f"  3/3  This is the last chance. Type exactly \"{code}\": ").strip() != code:
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
        if config.IMAGENT_HOME.name.lstrip(".").lower() != "imagent":  # (the folder is ~/.imagent) never delete a folder this program did not name itself
            problems.append("Did not delete the data folder: it is not named like Imagent's own folder.")
        elif _remove_tree(config.IMAGENT_HOME):
            print("  ✓ Deleted the data folder")
        else:
            problems.append("Could not delete everything in the data folder. Delete the folder yourself.")
    return problems


# --- Programs setup installed (asked about last) -----------------------------------------

def offer_tool_removal(tools: list[dict]) -> list[str]:
    """For each program setup installed, say what it is and ask whether to remove it too. Returns the problems.

    Removing one is deleting its folder in the tools folder, so nothing else on the computer can be touched."""
    problems = []
    for tool in tools:
        path = tool["path"]
        version = f" {tool['version']}" if tool["version"] else ""
        print(f"\n  {_bold(tool['name'] + version)}: {tool['description']}")
        print(f"  It takes {_fmt_bytes(_folder_size(path))} at {path}")
        if input(f"  Remove {tool['name']} too? [y/N]: ").strip().lower() not in ("y", "yes"):
            print(f"  {tool['name']} was kept. To remove it later, delete the folder {path}")
        elif path.resolve().parent != config.TOOLS_DIR.resolve():  # never delete a folder outside the tools folder
            problems.append(f"Did not remove {tool['name']}: its folder is not inside Imagent's tools folder. Delete {path} yourself.")
        elif _remove_tree(path):
            print(f"  ✓ Removed {tool['name']}")
        else:
            problems.append(f"Could not remove everything of {tool['name']}. Delete the folder {path} yourself.")
    try:
        config.TOOLS_DIR.rmdir()  # only succeeds when nothing is left in it
    except OSError:
        pass
    return problems


# --- Entry point -------------------------------------------------------------------------

def main() -> None:
    make_output_safe()
    if not sys.stdin.isatty():
        sys.exit("Uninstalling needs an interactive terminal. Run: imagent uninstall")

    stats = read_stats()
    host = config.load_config().get("ollama_host") or DEFAULT_HOST
    print("\n  Looking at what Imagent has added…")
    with _ollama(host) as base:
        plan = make_plan(base, stats)
        if nothing_to_remove(plan):
            if not plan["tools"]:
                print("\n  Nothing to uninstall: Imagent has not added anything to this computer (or it was already removed).")
                print(f"  {REMOVE_PROGRAM}")
                return
            print("\n  Imagent's data and models are already removed.")
            problems = []
        else:
            show_plan(stats, plan)
            if not confirm_three_times(stats):
                sys.exit(f"\n{CANCELLED}")
            if config.IMAGENT_HOME.exists():
                try:
                    with single_instance("indexing", "Indexing"):  # deleting under a running index would make it fail on every image
                        pass
                except AlreadyRunning as exc:
                    sys.exit(f"\n{exc} Nothing was deleted.")
            print()
            problems = delete_everything(plan, base)

    problems += offer_tool_removal(plan["tools"])  # after the block above, so a server started only for the models is stopped first

    print()
    if problems:
        print("  Finished, with problems:")
        for problem in problems:
            print(f"  ✗ {problem}")
    else:
        print("  ✓ Imagent is uninstalled.")
    print(f"  {REMOVE_PROGRAM}")
    sys.exit(1 if problems else 0)

