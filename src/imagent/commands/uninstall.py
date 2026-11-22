"""
imagent/commands/uninstall.py — Take Imagent off this computer: show what will be lost, ask three times, then delete.

Run by ``imagent uninstall``. It removes:
  * the Ollama models that setup downloaded or copied (listed in install_record.json); a model you already had
    before Imagent is never removed;
  * Imagent's data folder (the index database, config.json, logs);
  * Ollama itself, but only when setup installed it (listed in the plan and covered by the same three confirmations).
It never touches your image files or any other Ollama model, and leaves a pre-existing Ollama alone. The program itself is removed afterwards
with ``pipx uninstall imagent`` (a running program cannot remove itself), and the last message says so.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import shlex
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
LINUX_SERVICE_FILE = Path("/etc/systemd/system/ollama.service")  # where Ollama's Linux install script puts the service
LINUX_DATA_DIR = Path("/usr/share/ollama")  # what that service stores, models included
LINUX_BIN_DIRS = ("/usr/local/bin", "/usr/bin", "/bin")  # where the script may put the program
LINUX_DOCS = "https://github.com/ollama/ollama/blob/main/docs/linux.md"
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
            if _is_local(host) and shutil.which("ollama"):
                try:
                    process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               env={**os.environ, "OLLAMA_HOST": host},
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
        "ollama_installed": config.load_install_record().get("ollama_installed") is True,
        "ollama_command": None,
    }
    if plan["ollama_installed"]:
        plan["ollama_command"] = ollama_removal_command()
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
    if plan["ollama_installed"]:
        command = plan["ollama_command"]
        if command is None:
            print("\n  Imagent installed Ollama itself, and it will not be removed. Remove it the way you would remove any program"
                  + (f" (on Linux: see the Uninstall section of {LINUX_DOCS})." if platform.system() == "Linux" else "."))
        else:
            print("\n  Imagent installed Ollama itself, so Ollama is removed too.")
            if platform.system() == "Linux":
                print(f"  That stops the Ollama service and deletes Ollama, its \"ollama\" user and everything in {LINUX_DATA_DIR},")
                print("  including models you downloaded yourself. It asks for your password (sudo).")
            print(f"  Command: {shlex.join(command)}")
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
        if config.IMAGENT_HOME.name.lower() != "imagent":  # never delete a folder this program did not name itself
            problems.append("Did not delete the data folder: it is not named like Imagent's own folder.")
        elif _remove_tree(config.IMAGENT_HOME):
            print("  ✓ Deleted the data folder")
        else:
            problems.append("Could not delete everything in the data folder. Delete the folder yourself.")
    return problems


# --- Ollama itself (only when setup installed it) --------------------------------------

def ollama_removal_command() -> list[str] | None:
    """The one command that removes the Ollama that setup installed, or None when this computer does not look the way
    setup left it (then the user is told how to remove it by hand). Linux has no uninstall command, so the documented
    steps are joined into one ``sudo sh -c`` line; it ends by checking that the program and the service are gone."""
    system = platform.system()
    if system == "Darwin" and shutil.which("brew"):
        return ["brew", "uninstall", "ollama"]
    if system == "Windows" and shutil.which("winget"):
        return ["winget", "uninstall", "--id", "Ollama.Ollama", "-e", "--silent"]
    if system != "Linux" or not shutil.which("systemctl") or not LINUX_SERVICE_FILE.exists():
        return None
    program = shutil.which("ollama")
    if program is None or str(Path(program).parent) not in LINUX_BIN_DIRS:
        return None
    sudo = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo"]
    if sudo and not shutil.which("sudo"):
        return None
    libraries = Path(program).parent.parent / "lib" / "ollama"
    paths = [str(LINUX_SERVICE_FILE), program]
    folders = [str(LINUX_DATA_DIR)] + ([str(libraries)] if libraries.is_dir() else [])
    steps = ["systemctl stop ollama", "systemctl disable ollama",
             "rm -f " + " ".join(map(shlex.quote, paths)), "rm -rf " + " ".join(map(shlex.quote, folders)),
             "userdel ollama", "groupdel ollama 2>/dev/null",
             "test ! -e " + shlex.quote(str(LINUX_SERVICE_FILE)) + " && test ! -e " + shlex.quote(program)]
    return sudo + ["sh", "-c", "; ".join(steps)]


def remove_ollama(command: list[str]) -> list[str]:
    """Run the command that removes the Ollama that setup installed. Returns the problems."""
    if subprocess.run(command).returncode == 0:
        print("  ✓ Removed Ollama")
        return []
    return [f"Could not remove Ollama. Remove it yourself with: {shlex.join(command)}"]


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
            print("\n  Nothing to uninstall: Imagent has not added anything to this computer (or it was already removed).")
            print(f"  {REMOVE_PROGRAM}")
            return
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

    if plan["ollama_command"]:  # after the block above, so a server started only for the models is stopped first
        problems += remove_ollama(plan["ollama_command"])

    print()
    if problems:
        print("  Finished, with problems:")
        for problem in problems:
            print(f"  ✗ {problem}")
    else:
        print("  ✓ Imagent is uninstalled.")
    print(f"  {REMOVE_PROGRAM}")
    sys.exit(1 if problems else 0)

