#!/usr/bin/env python3
"""Interactive first-run setup: show system details, check packages and Ollama,
collect model names, save ~/.imagent/config.json. Never starts Ollama or any model."""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
from contextlib import contextmanager
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
IMAGENT_HOME = Path(os.getenv("IMAGENT_HOME", str(Path.home() / ".imagent")))
CONFIG_PATH = IMAGENT_HOME / "config.json"
DEFAULT_HOST = "http://localhost:11434"
RELAUNCHED_ENV = "IMAGENT_SETUP_RELAUNCHED"
REGISTRY = "https://registry.ollama.ai/v2"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
RED = "fg:ansibrightred"
TOTAL_STEPS = 5
INDENT = "  "  # left padding for everything under a step heading
QMARK = INDENT + "?"
POINTER = INDENT + "»"

ASK_AI = "Not sure? Ask AI. Copy the prompt and give it to your favorite LLM."
COPIED = "✓ Prompt copied to clipboard. Paste it into your favorite LLM. (Select again to copy once more)"

questionary = None  # imported in step 2, once the packages are available
STYLE = None  # prompt colours, built together with the questionary import


# --- Output helpers -----------------------------------------------------------

def say(message: str = "", style: str = "") -> None:
    message = "\n".join(INDENT + line if line else line for line in message.split("\n"))
    if questionary is not None:
        questionary.print(message, style=style)
    else:
        print(message)


def step(number: int, title: str) -> None:
    os.system("")  # enables ANSI colours in Windows terminals
    print(f"\n\033[1m▌Step {number} of {TOTAL_STEPS}: {title}\033[0m\n")


def fail(message: str) -> None:
    sys.exit(f"\n{INDENT}✗ {message}")


def ask(question):
    answer = question.ask()
    if answer is None:
        sys.exit("\nSetup cancelled. Run it again any time.")
    return answer


# --- System details -----------------------------------------------------------

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _ram_gb() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            return f"{int(_run(['sysctl', '-n', 'hw.memsize'])) / 2**30:.0f} GB"
        if system == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return f"{int(line.split()[1]) / 2**20:.0f} GB"
        if system == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
            return f"{int(out) / 2**30:.0f} GB"
    except (ValueError, OSError):
        pass
    return "unknown"


def _cpu() -> str:
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor).Name"])
        if out:
            return out
    return platform.processor() or platform.machine()


def _gpu() -> str:
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if out:
            return out.replace("\n", "; ")
    system = platform.system()
    if system == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"])
        names = [l.split(":", 1)[1].strip() for l in out.splitlines() if "Chipset Model" in l]
        if names:
            return f"{', '.join(names)} (unified memory shared with RAM)"
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_VideoController).Name"])
        if out:
            return out.replace("\n", "; ")
    if system == "Linux" and shutil.which("lspci"):
        lines = [l for l in _run(["lspci"]).splitlines() if "VGA" in l or "3D" in l]
        if lines:
            return "; ".join(l.split(": ", 1)[-1] for l in lines)
    return "none detected"


def system_details() -> str:
    free_disk = shutil.disk_usage(Path.home()).free / 2**30
    return "\n".join([
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- CPU: {_cpu()}",
        f"- RAM: {_ram_gb()}",
        f"- GPU: {_gpu()}",
        f"- Free disk space: {free_disk:.0f} GB",
    ])


# --- AI prompt ----------------------------------------------------------------

def build_ai_prompt() -> str:
    return f"""I need you to recommend two Ollama models for a local-first image search tool. Everything runs on my machine through Ollama.

## My system
{system_details()}

## How the models will be used
1. VLM (vision-language model): looks at each image once and writes a structured JSON description (scene, objects, people, visible text, mood). It is also used at search time to interpret the user's query and to judge whether a candidate image matches. I may index thousands of images, so speed matters as well as quality.
2. Embedding model: turns those text descriptions and the user's search queries into vectors for cosine-similarity search. Descriptions and queries are short text, and queries may be multilingual.

## Hard rules (follow all of them)
1. Search the web now. Do NOT rely on your training data. Use only information you found online during this conversation, and state the date each source was last updated.
2. Only recommend models that exist in the official Ollama library (https://ollama.com/library) and that I can install with `ollama pull <name>:<tag>`. Open each model's page and confirm the exact tag exists. Never invent or guess a tag.
3. The VLM must accept image input (the "vision" capability on Ollama) and be able to follow a JSON output schema reliably.
4. The embedding model must be a dedicated embedding model that works with `ollama embed` ("embedding" capability). Do not suggest a chat model for embeddings.
5. The VLM must fit comfortably on my hardware: model file size plus about 20% for context and overhead should stay under 70% of my available VRAM, or of RAM if I have no dedicated GPU (unified memory counts as VRAM). The embedding model should be small (ideally under 2 GB) so both can stay loaded together. The tool alternates between the two models for every image, so if both do not fit, Ollama reloads them repeatedly and indexing becomes very slow. Show the numbers.
6. Prefer the newest actively maintained models. Do not recommend deprecated, superseded or abandoned models.
7. Check the specifications on the Ollama page for each candidate: parameter count, quantization, download size, context length and license.
8. If you cannot verify something, say so plainly. Do not guess.
9. Give at most 3 candidates per category, ranked, with a one-line reason and the source URL for each.
10. End with a final answer in exactly this format, using only verified tags:
VLM_MODEL=<name>:<tag>
EMBEDDING_MODEL=<name>:<tag>
"""


def copy_to_clipboard(text: str) -> bool:
    system = platform.system()
    if system == "Darwin":
        candidates = [["pbcopy"]]
    elif system == "Windows":
        candidates = [["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"]]
    else:
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for cmd in candidates:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text, text=True, encoding="utf-8", check=True, timeout=10)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


def offer_prompt_copy() -> bool:
    """Copy the AI prompt to the clipboard. If that fails, print it for manual copying."""
    prompt = build_ai_prompt()
    if copy_to_clipboard(prompt):
        return True
    say("Could not reach your clipboard. Select and copy the text below by hand,", "fg:ansimagenta")
    say("then paste it into any AI chat:\n", "fg:ansimagenta")
    print(prompt)
    return False


# --- Step 2: packages ---------------------------------------------------------

def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def _pip_install(python: str) -> None:
    cmd = [python, "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)]
    if subprocess.run(cmd).returncode != 0:
        fail("Could not install the required packages. Check your internet connection and run setup again.")


def ensure_packages() -> None:
    """Make sure packages live in a virtual environment; relaunch inside .venv if needed."""
    global questionary, STYLE
    if not _in_venv():
        if not _venv_python().exists():
            say("Creating a private Python environment in .venv (one time only)…")
            try:
                venv.create(VENV_DIR, with_pip=True)
            except Exception as exc:
                hint = "  On Ubuntu/Debian try: sudo apt install python3-venv" if platform.system() == "Linux" else ""
                fail(f"Could not create the .venv environment ({exc}).\n{hint}")
        say("Installing required packages into .venv (this can take a minute)…")
        _pip_install(str(_venv_python()))
        result = subprocess.run(
            [str(_venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]],
            env={**os.environ, RELAUNCHED_ENV: "1"},
        )
        sys.exit(result.returncode)
    try:
        import questionary as q
    except ImportError:
        say("Installing required packages (this can take a minute)…")
        _pip_install(sys.executable)
        import questionary as q
    questionary = q
    STYLE = q.Style([
        ("qmark", "fg:ansidefault bold"),
        ("question", "fg:ansidefault bold"),
        ("pointer", "fg:ansidefault bold"),
        ("highlighted", "fg:ansidefault bold"),
        ("selected", "noreverse fg:ansidefault bold"),
        ("answer", "fg:ansidefault bold"),
        ("instruction", "fg:ansibrightblack"),
        ("disabled", "fg:ansibrightblack italic"),
    ])
    say("✓ Required packages are installed.", "fg:green")


# --- Step 3: Ollama -----------------------------------------------------------

def install_command():
    system = platform.system()
    if system == "Darwin" and shutil.which("brew"):
        return ["brew", "install", "ollama"]
    if system == "Linux":
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    if system == "Windows" and shutil.which("winget"):
        return ["winget", "install", "--id", "Ollama.Ollama", "-e"]
    return None


def ensure_ollama_installed() -> None:
    if shutil.which("ollama"):
        say("✓ Ollama is installed.", "fg:green")
        return
    say("Ollama is the program that runs the AI models on your computer. It is not installed yet.", "fg:ansimagenta")
    cmd = install_command()
    if cmd is None:
        fail("Please install Ollama from https://ollama.com/download, then run setup again.")
    say(f"Install command: {cmd[-1] if cmd[0] == 'sh' else ' '.join(cmd)}")
    if not ask(questionary.confirm("Install Ollama now?", default=True, qmark=QMARK, style=STYLE)):
        fail("Ollama is required. Install it from https://ollama.com/download, then run setup again.")
    if subprocess.run(cmd).returncode != 0:
        fail("The Ollama install did not finish. Install it from https://ollama.com/download, then run setup again.")
    if shutil.which("ollama"):
        say("✓ Ollama installed.", "fg:green")
    else:
        say("✓ Ollama installed. Open a new terminal window afterwards so the `ollama` command works.", "fg:green")


# --- Step 4: models -----------------------------------------------------------

def _clean_model_name(raw: str) -> str:
    """Accept pasted 'VLM_MODEL=name:tag' lines as well as plain names."""
    return raw.strip().split("=", 1)[-1].strip().strip("`'\"")


def _validate_model(value: str):
    name = _clean_model_name(value)
    if not name:
        return "Please paste a model name, for example: name:tag"
    if any(c.isspace() for c in name):
        return "A model name has no spaces. Paste only the name, for example: name:tag"
    return True


def _ask_name(label: str, default: str = "") -> str:
    """Ask for a model name. Returns '' when the box is left empty (clears the field)."""
    def validate(value: str):
        return True if not value.strip() else _validate_model(value)
    return _clean_model_name(ask(questionary.text(
        label, validate=validate, default=default, qmark=QMARK, style=STYLE,
        instruction="(paste it here, leave empty to clear) ")))


def _erase_last_line() -> None:
    """Remove the answered prompt line so the next menu redraws in the same place."""
    print("\033[1A\033[2K", end="", flush=True)


def _field_choice(label: str, value: str, key: str):
    """Menu row for a model field: green once filled, light red while unset."""
    if value:
        return questionary.Choice([("fg:ansigreen", f"✓ {label} {value}")], value=key)
    return questionary.Choice([("fg:ansibrightred", f"○ {label} not set")], value=key)


def ask_models(current: dict):
    """One menu: open any field, copy the AI prompt, or continue once both names are set."""
    vlm, embedding = current.get("vlm_model", ""), current.get("embedding_model", "")
    say("Fill in both model names, then choose Continue.\n")
    copied = False
    while True:
        ready = bool(vlm and embedding)
        choices = [
            _field_choice("VLM model:      ", vlm, "vlm"),
            _field_choice("Embedding model:", embedding, "embedding"),
            questionary.Separator(),
            questionary.Choice([("fg:ansigreen", COPIED)] if copied else ASK_AI, value="ask"),
            questionary.Separator(),
            questionary.Choice("Continue", value="continue",
                               disabled=None if ready else "enter both model names first"),
        ]
        default = "continue" if ready else ("embedding" if vlm else "vlm")
        choice = ask(questionary.select(
            "What would you like to do?", choices=choices, default=default,
            qmark=QMARK, pointer=POINTER, style=STYLE))
        if choice == "continue":
            return vlm, embedding
        _erase_last_line()
        if choice == "ask":
            copied = offer_prompt_copy()
        elif choice == "vlm":
            vlm = _ask_name("VLM model name:", vlm)
            _erase_last_line()
        else:
            embedding = _ask_name("Embedding model name:", embedding)
            _erase_last_line()


# --- Model lookup (ollama.com registry, no Ollama server needed) ---------------

class ModelNotFound(Exception):
    pass


class RegistryUnreachable(Exception):
    pass


def _fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def _split_model(name: str):
    """'name[:tag]' or 'user/name[:tag]' -> (repository, tag). None for external registries (e.g. hf.co/...)."""
    repo, _, tag = name.partition(":")
    if "/" in repo and "." in repo.split("/")[0]:
        return None
    return (repo if "/" in repo else f"library/{repo}"), tag or "latest"


def _registry_get(path: str, accept: str = "application/json") -> dict:
    request = urllib.request.Request(f"{REGISTRY}/{path}", headers={"Accept": accept, "User-Agent": "imagent-setup"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ModelNotFound() from exc
        raise RegistryUnreachable(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RegistryUnreachable(str(exc)) from exc


def fetch_model_specs(name: str) -> dict:
    """Look the model up on ollama.com. Raises ModelNotFound / RegistryUnreachable."""
    parsed = _split_model(name)
    if parsed is None:
        return {"rows": [("Details", "not checked (model from another registry)")], "bytes": 0}
    repo, tag = parsed
    manifest = _registry_get(f"{repo}/manifests/{tag}", MANIFEST_ACCEPT)
    try:
        size = manifest["config"]["size"] + sum(layer["size"] for layer in manifest["layers"])
        digest = manifest["config"]["digest"]
    except (KeyError, TypeError) as exc:
        raise RegistryUnreachable("unexpected response") from exc
    rows = [("Download size", _fmt_bytes(size))]
    try:
        config = _registry_get(f"{repo}/blobs/{digest}")
        for label, key in (("Parameters", "model_type"), ("Quantization", "file_type"), ("Family", "model_family")):
            if config.get(key):
                rows.append((label, str(config[key])))
    except RegistryUnreachable:
        pass  # size alone is enough to continue
    return {"rows": rows, "bytes": size}


def review_models(vlm: str, embedding: str):
    """Verify both models exist, show their specs, then ask Continue / Go back.

    Returns (proceed, names_not_found)."""
    say("\nChecking the models on ollama.com…")
    specs, problems, missing = {}, [], set()
    for name in dict.fromkeys([vlm, embedding]):
        try:
            specs[name] = fetch_model_specs(name)
        except ModelNotFound:
            missing.add(name)
            problems.append(f"✗ '{name}' was not found on ollama.com. Check the name and try again.")
        except RegistryUnreachable:
            problems.append("✗ Could not reach ollama.com. Check your internet connection and try again.")
            break
    if problems:
        say("")
        for problem in problems:
            say(problem, RED)
        say("")
        return False, missing

    say("")
    for label, name in (("VLM model", vlm), ("Embedding model", embedding)):
        say(f"{label}: {name}", "bold")
        for key, value in specs[name]["rows"]:
            say(f"    {key + ':':<15}{value}")
        say("")
    total = sum(specs[name]["bytes"] for name in dict.fromkeys([vlm, embedding]))
    free = shutil.disk_usage(Path.home()).free
    say(f"Total download: {_fmt_bytes(total)}    Free disk space: {_fmt_bytes(free)}\n", "bold")
    enough_space = total * 1.1 <= free
    if not enough_space:
        say("✗ Not enough free disk space for these models. Free up space or go back and choose smaller models.\n", RED)

    choice = ask(questionary.select(
        "Ready to download?",
        choices=[
            questionary.Choice("← Go back and change the models", value="back"),
            questionary.Separator(),
            questionary.Choice("Continue: download the models", value="go",
                               disabled=None if enough_space else "not enough disk space"),
        ],
        default="go" if enough_space else "back",
        qmark=QMARK, pointer=POINTER, style=STYLE,
    ))
    return choice == "go", missing


# --- Step 5: download ---------------------------------------------------------

def _is_local(host: str) -> bool:
    return any(h in host for h in ("localhost", "127.0.0.1", "[::1]"))


def _pull_with_progress(client, name: str) -> None:
    last = ""
    for progress in client.pull(name, stream=True):
        line = progress.status or ""
        if progress.total and progress.completed is not None:
            line = f"{line}  {progress.completed * 100 // progress.total}%  ({_fmt_bytes(progress.completed)} of {_fmt_bytes(progress.total)})"
        if line != last:
            print(f"\r\033[K{INDENT}{line}", end="", flush=True)
            last = line
    print("\r\033[K", end="", flush=True)


def _start_server():
    kwargs = {"creationflags": 0x00000008} if platform.system() == "Windows" else {"start_new_session": True}
    return subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)


def _stop_server(process) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    say("Stopped the temporary Ollama server.")


@contextmanager
def ollama_client(host: str):
    """Yield a connected Ollama client, starting `ollama serve` only if needed and stopping it afterwards.

    Yields None (after printing the reason) when Ollama cannot be reached or started."""
    import ollama
    client = ollama.Client(host=host)

    def reachable() -> bool:
        try:
            client.list()
            return True
        except Exception:
            return False

    started = None
    try:
        if not reachable():
            if not _is_local(host):
                say(f"✗ Cannot reach Ollama at {host}.", RED)
                yield None
                return
            if not shutil.which("ollama"):
                say("✗ The `ollama` command was not found. Open a new terminal window and run setup again.", RED)
                yield None
                return
            say("Starting Ollama in the background…")
            started = _start_server()
            for _ in range(30):
                if reachable():
                    break
                time.sleep(1)
            else:
                say("✗ Ollama did not start. Open the Ollama app (or run `ollama serve`) and try again.", RED)
                yield None
                return
        yield client
    finally:
        if started is not None:
            _stop_server(started)


def _normalize(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def _pull_models(client, models: list) -> None:
    """Pull each model that is not installed yet, showing progress."""
    installed = {_normalize(m.model) for m in client.list().models}
    for name in dict.fromkeys(models):
        if _normalize(name) in installed:
            say(f"✓ {name} is already downloaded.", "fg:ansigreen")
            continue
        say(f"Downloading {name}…")
        _pull_with_progress(client, name)
        say(f"✓ {name} downloaded.", "fg:ansigreen")


def download_models(host: str, models: list) -> bool:
    with ollama_client(host) as client:
        if client is None:
            return False
        try:
            _pull_models(client, models)
            return True
        except Exception as exc:
            print()
            say(f"✗ Download failed: {exc}", RED)
            return False


# --- Re-run: show installed models --------------------------------------------

def show_installed_models(config: dict) -> None:
    """Setup was run before: list every model installed in Ollama, marking the ones in use."""
    say("Setup was already completed. Checking your installed models…\n")
    roles = {_normalize(config["vlm_model"]): "VLM", _normalize(config["embedding_model"]): "Embedding"}
    with ollama_client(config["ollama_host"]) as client:
        if client is None:
            sys.exit(1)
        try:
            installed = {_normalize(m.model): m.size or 0 for m in client.list().models}
        except Exception as exc:
            fail(f"Could not read the model list from Ollama: {exc}")

        in_use = [n for n in roles if n in installed]
        missing = [n for n in roles if n not in installed]
        others = sorted(n for n in installed if n not in roles)
        width = max(len(n) for n in [*in_use, *missing, *others, ""])

        say("Installed models:", "bold")
        for name in in_use:
            say(f"✓ {name:<{width}}  {_fmt_bytes(installed[name]):>8}  {roles[name]} (in use)", "fg:ansigreen")
        for name in missing:
            say(f"✗ {name:<{width}}  {'':>8}  {roles[name]} (in use, NOT installed)", RED)
        for name in others:
            say(f"  {name:<{width}}  {_fmt_bytes(installed[name]):>8}")
        if not installed:
            say("  (no models installed)")
        say("\nTo change the models you use, run change_model.py.")

        if missing and ask(questionary.confirm(
                "\nDownload the missing model(s) now?", default=True, qmark=QMARK, style=STYLE)):
            try:
                _pull_models(client, [config["vlm_model"], config["embedding_model"]])
            except Exception as exc:
                print()
                fail(f"Download failed: {exc}")


# --- Main ---------------------------------------------------------------------

def load_existing_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main() -> None:
    if not sys.stdin.isatty():
        fail("Setup needs an interactive terminal. Run it directly: python3 setup.py")

    if not os.getenv(RELAUNCHED_ENV):
        step(1, "Your computer")
        say(system_details())
        step(2, "Required packages")
    ensure_packages()

    step(3, "Ollama")
    ensure_ollama_installed()

    current = load_existing_config()
    if all(current.get(k) for k in ("ollama_host", "vlm_model", "embedding_model")):
        global TOTAL_STEPS
        TOTAL_STEPS = 4
        step(4, "Installed models")
        show_installed_models(current)
        return

    step(4, "Models")
    vlm_model, embedding_model = current.get("vlm_model", ""), current.get("embedding_model", "")
    host = None
    while True:
        vlm_model, embedding_model = ask_models({"vlm_model": vlm_model, "embedding_model": embedding_model})
        proceed, missing = review_models(vlm_model, embedding_model)
        if not proceed:
            # Clear only the names that were not found; keep the valid one.
            vlm_model = "" if vlm_model in missing else vlm_model
            embedding_model = "" if embedding_model in missing else embedding_model
            step(4, "Models")
            continue
        step(5, "Download")
        if host is None:
            host = ask(questionary.text(
                "Where is Ollama running? Press Enter if it is on this computer:",
                default=current.get("ollama_host", DEFAULT_HOST), qmark=QMARK, style=STYLE,
            )).strip() or DEFAULT_HOST
            say("")
        if download_models(host, [vlm_model, embedding_model]):
            break
        step(4, "Models")

    IMAGENT_HOME.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"ollama_host": host, "vlm_model": vlm_model, "embedding_model": embedding_model}, indent=2) + "\n",
        encoding="utf-8",
    )
    say(f"\n✓ Setup complete. Your choices are saved in {CONFIG_PATH}", "fg:green")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nSetup cancelled. Run it again any time.")
