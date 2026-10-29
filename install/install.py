#!/usr/bin/env python3
"""One-command installer and first-run setup: show system details, create .venv and check its
packages, check Python and Ollama versions, collect and verify the model names, then lock them in the
Imagent database. Anything that is missing or too old is listed and installed only after you agree,
and packages only ever go into .venv. The models cannot be changed afterwards; running this again only
shows their status and reinstalls a missing one. Starts Ollama only temporarily, to download the models."""

import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from contextlib import contextmanager
from pathlib import Path

if sys.version_info < (3, 9):  # before the project's own modules, which need 3.9 just to be imported
    _how = {"Darwin": "brew install python", "Windows": "winget install Python.Python.3.12"}.get(
        platform.system(), "sudo apt install python3 python3-venv (or your package manager)")
    _installer = "install.bat" if platform.system() == "Windows" else "sh install.sh"
    sys.exit(f"Imagent needs Python 3.9 or newer, and this is Python {platform.python_version()}. "
             f"Install a newer one (for example: {_how}, or https://www.python.org/downloads/), "
             f"then run it again: {_installer}")

# Make the project root importable when this file is run directly (``python install/install.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as app_config
from db.connection import init_db
from db.operations import get_model_settings, set_model_settings
from pipeline.console import make_output_safe, run_cli

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
CONFIG_PATH = app_config.CONFIG_PATH
DEFAULT_HOST = "http://localhost:11434"
RELAUNCH_FLAG = "--relaunched"  # passed to the copy of this script that runs inside .venv
INSTALL_TIMEOUT = 1800  # seconds allowed for pip and for installing Ollama
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9._-]+)?$")
REGISTRY = "https://registry.ollama.ai/v2"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
RED = "fg:ansibrightred"
INDENT = "  "  # left padding for everything under a step heading
QMARK = INDENT + "?"
POINTER = INDENT + "»"

ASK_AI = "Not sure? Ask AI. Copy the prompt and give it to your favorite LLM."
COPIED = "✓ Prompt copied to clipboard. Paste it into your favorite LLM. (Select again to copy once more)"
PASTE_SET = "Paste the AI's answer: one set of three models (vision, text, embedding)."

questionary = None  # imported in step 2, once the packages are available
STYLE = None  # prompt colours, built together with the questionary import


# --- Output helpers -----------------------------------------------------------

def say(message: str = "", style: str = "") -> None:
    message = "\n".join(INDENT + line if line else line for line in message.split("\n"))
    if questionary is not None:
        questionary.print(message, style=style)
    else:
        print(message)


def step(title: str) -> None:
    os.system("")  # enables ANSI colours in terminals
    print(f"\n\033[1m▌{title}\033[0m\n")


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
    return f"""I need to choose AI models for a local image search tool that runs on my computer through Ollama. Balance speed and accuracy, but prefer accuracy whenever the speed gain is small.

## My computer
{system_details()}

## The three models
- Vision model: looks at each image once and writes a JSON description (scene, objects, people, colors and mood, tags). A separate OCR step reads text in images. The descriptions must be accurate, and I may index thousands of images, so speed matters too.
- Text model: fixes typos in my search query and writes a short description of the image I am looking for. It never sees images. One request per search.
- Embedding model: turns the stored descriptions (about 500 tokens each) and my search queries (possibly in several languages) into vectors. Used when indexing and when searching.

## Rules
1. Search the web now. Do not rely on your memory.
2. Only models from https://ollama.com/library, each with an exact tag that you confirmed on its page so that `ollama pull <name>:<tag>` works. Never guess or invent a tag.
3. Capabilities: the vision model needs "vision", the text model needs "completion", the embedding model needs "embedding" (context of at least 1024 tokens, strong multilingual retrieval). The vision and text models must follow a JSON schema reliably and must not be thinking-only (reasoning) models. The text model may be the same model as the vision model.
4. Quantization of 4-bit or better, never lower.
5. Speed: a vision reply within a few seconds per image (never more than about 30 seconds) and a text reply within about 10 seconds. If two models are within about 30% of each other in speed, choose the more accurate one. Choose the faster one only when it is clearly faster (about twice as fast) and only slightly less accurate. Judge by published benchmarks and the model's own page, not by size, popularity or release date; an older maintained model is fine if the balance is better.
6. Memory: model weights are fully loaded (no memory-mapping). Every set must fit entirely on the GPU, with no CPU offload and no swapping, in about 85% of my VRAM (or RAM if there is no dedicated GPU; with shared memory, the part the GPU can use). Check both cases; the vision and text models are never loaded together:
   - indexing: vision model + embedding model, including the vision model's KV cache at a context of {app_config.OLLAMA_NUM_CTX} tokens and its image-encoder working memory for images up to {app_config.IMAGE_MAX_LONG_EDGE} px on the long edge (reserved for a worst-case image);
   - searching: text model + embedding model, including the text model's KV cache at a context of {app_config.OLLAMA_NUM_CTX} tokens.
7. Do the memory sums yourself and do not show the working.

## Answer
Give three sets. A set is one complete choice: a vision model, a text model and an embedding model that work together. Set 1 is the best balance and the one you recommend; set 2 is a faster or lighter alternative; set 3 is a more accurate one that still fits. Only give a set if you verified all three of its models; if there are fewer than three, give fewer and say so.

Reply with exactly these three parts, in this order, and nothing else:
1. A table with one row per set and the columns: Set, Vision model, Text model, Embedding model, Memory (indexing / searching), Speed, Accuracy (the scores you used).
2. One or two sentences per set on the trade-offs against the other two sets: what I gain and what I give up.
3. The heading "Copy one set", then one code block per set (plain, no language name). Each code block holds one line and nothing else: the three model names, in the order vision, text, embedding, separated by commas.
   - Exact shape: visionname:7b,textname:8b,embeddingname:335m
   - Forbidden in that line: spaces, labels, numbering, bullets, quotes, brackets, JSON and line breaks.
   - Before you reply, check that every such line has exactly two commas and three `name:tag` values.
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

def _in_project_venv() -> bool:
    """True only inside this project's .venv; another virtual environment (or conda) the user has active does not count."""
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def _confirm(question: str) -> bool:
    """Plain yes/no question for the steps that run before the packages (and so questionary) exist."""
    return input(f"{INDENT}{question} [Y/n] ").strip().lower() in ("", "y", "yes")


def _succeeds(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, timeout=INSTALL_TIMEOUT).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _venv_works(python: Path) -> bool:
    """False when .venv exists but its Python is gone, broken or older than 3.9."""
    try:
        out = subprocess.run([str(python), "-c", "import sys; print(sys.version_info >= (3, 9))"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == "True"


def _requirements() -> list:
    """(package name, minimum version or None) for each line of requirements.txt."""
    found = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:>=\s*([0-9][0-9.]*))?", line)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def _version_tuple(text: str) -> tuple:
    numbers = [int(part) for part in re.findall(r"\d+", text.split("+")[0])[:3]]
    return tuple(numbers + [0] * (3 - len(numbers)))


def _installed_versions(python: str, names: list):
    """{name: installed version or None} as seen by ``python``; None when it cannot be told."""
    code = ("import json, sys\nfrom importlib import metadata\nfound = {}\nfor name in sys.argv[1:]:\n"
            "    try:\n        found[name] = metadata.version(name)\n    except metadata.PackageNotFoundError:\n"
            "        found[name] = None\nprint(json.dumps(found))")
    try:
        out = subprocess.run([python, "-I", "-c", code, *names], capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _pip_install(python: str) -> None:
    cmd = [python, "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)]
    if not _succeeds(cmd):
        fail("Could not install the required packages (the messages above say which one). Check your internet connection and "
             f"run {app_config.INSTALL_COMMAND} again. If a package has no version for your Python ({platform.python_version()}), try another Python, "
             "such as 3.11 or 3.12.")


def _sync_packages(python: str) -> None:
    """Install into .venv what requirements.txt needs and it lacks or has too old, after asking. Nothing outside .venv changes."""
    wanted = _requirements()
    installed = _installed_versions(python, [name for name, _ in wanted])
    todo = []
    for name, minimum in wanted:
        have = (installed or {}).get(name)
        if have is None:
            todo.append(f"{name} (not installed)")
        elif minimum and _version_tuple(have) < _version_tuple(minimum):
            todo.append(f"{name} {have} is older than {minimum}")
    if installed is not None and not todo:
        return
    if installed is None:
        say("Could not read the installed packages, so every required package will be checked and installed.")
    else:
        say("These packages are needed in .venv:")
        for item in todo:
            say(f"  - {item}")
    say("They go into .venv only; Python and the packages elsewhere on your computer are not touched.")
    if not _confirm("Install them now? (this can take a minute)"):
        fail(f"The packages are required. Run {app_config.INSTALL_COMMAND} again when you are ready.")
    _pip_install(python)


def _prepare_venv() -> None:
    """Create .venv if needed (or recreate a broken one, after asking) and bring its packages up to date."""
    python = _venv_python()
    if python.exists() and not _venv_works(python):
        say(".venv exists but its Python does not work (it was probably made by a Python that has since been removed or updated).")
        if not _confirm("Delete .venv and create it again?"):
            fail(f"Delete the .venv folder yourself, then run {app_config.INSTALL_COMMAND} again.")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    if not python.exists():
        say("Creating a private Python environment in .venv (one time only)…")
        try:
            venv.create(VENV_DIR, with_pip=True)
        except Exception as exc:
            hint = "  On Ubuntu/Debian try: sudo apt install python3-venv" if platform.system() == "Linux" else ""
            fail(f"Could not create the .venv environment ({exc}).\n{hint}")
    _sync_packages(str(python))


def ensure_packages() -> None:
    """Make sure packages live in the project's virtual environment; relaunch inside .venv if needed."""
    global questionary, STYLE
    if not _in_project_venv():
        if RELAUNCH_FLAG in sys.argv:  # the relaunch did not end up inside .venv; stop rather than loop
            fail(f"Could not start inside .venv. Delete the .venv folder and run {app_config.INSTALL_COMMAND} again.")
        _prepare_venv()
        result = subprocess.run([str(_venv_python()), str(Path(__file__).resolve()), RELAUNCH_FLAG, *sys.argv[1:]])
        sys.exit(result.returncode)
    try:
        import questionary as q
    except ImportError:
        _sync_packages(sys.executable)
        import questionary as q
    questionary = q
    STYLE = q.Style([
        ("qmark", "fg:ansidefault bold"),
        ("question", "fg:ansidefault bold"),
        ("pointer", "fg:ansidefault bold"),
        ("highlighted", "fg:ansidefault bold underline"),
        ("selected", "noreverse fg:ansidefault bold underline"),
        ("answer", "fg:ansidefault bold"),
        ("instruction", "fg:ansibrightblack"),
        ("disabled", "fg:ansibrightblack italic"),
    ])
    say("✓ Required packages are installed.", "fg:green")


def prepare_ocr() -> None:
    """Download the OCR models now, while someone is watching, so that indexing can run offline later."""
    say("Preparing the text-reading (OCR) models; the first time this downloads them…")
    try:
        from image_processing.ocr import check_available
        check_available()
    except Exception as exc:
        say(f"⚠ Could not prepare them ({type(exc).__name__}: {exc}).", "fg:ansimagenta")
        say("Indexing tries again, and needs an internet connection the first time.", "fg:ansimagenta")
        return
    say("✓ Text reading is ready.", "fg:green")


# --- Step 3: Ollama -----------------------------------------------------------

def install_command(upgrade: bool = False):
    system = platform.system()
    if system == "Darwin" and shutil.which("brew"):
        return ["brew", "upgrade" if upgrade else "install", "ollama"]
    if system == "Linux":
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]  # also updates an existing install
    if system == "Windows" and shutil.which("winget"):
        return ["winget", "upgrade" if upgrade else "install", "--id", "Ollama.Ollama", "-e"]
    return None


def _ollama_command_version():
    """Version of the installed `ollama` command as numbers, e.g. (0, 6, 4); None if it cannot be told."""
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout + out.stderr
    match = re.search(r"client version is (\d+)\.(\d+)\.(\d+)", text) or re.search(r"version is (\d+)\.(\d+)\.(\d+)", text)
    version = tuple(int(part) for part in match.groups()) if match else None
    return version if version and any(version) else None  # a development build reports 0.0.0


def ensure_ollama_current() -> None:
    """Offer to update a local Ollama older than Imagent needs. An Ollama on another computer is checked once its address is known."""
    from vlm.client import MIN_OLLAMA_VERSION  # needs the packages that ensure_packages installed
    version = _ollama_command_version()
    if version is None or version >= MIN_OLLAMA_VERSION:
        return
    needed = ".".join(map(str, MIN_OLLAMA_VERSION))
    say(f"Your Ollama is version {'.'.join(map(str, version))}, but Imagent needs {needed} or newer.", "fg:ansimagenta")
    cmd = install_command(upgrade=True)
    if cmd is None:
        fail(f"Please update Ollama from https://ollama.com/download, then run {app_config.INSTALL_COMMAND} again.")
    say(f"Update command: {cmd[-1] if cmd[0] == 'sh' else ' '.join(cmd)}")
    if not ask(questionary.confirm("Update Ollama now?", default=True, qmark=QMARK, style=STYLE)):
        fail(f"Imagent needs Ollama {needed} or newer. Update it from https://ollama.com/download, then run {app_config.INSTALL_COMMAND} again.")
    if not _succeeds(cmd):
        fail(f"The Ollama update did not finish. Update it from https://ollama.com/download, then run {app_config.INSTALL_COMMAND} again.")
    say("✓ Ollama updated. If the Ollama app is open, quit and reopen it so it uses the new version.", "fg:green")


def ensure_ollama_installed() -> None:
    if shutil.which("ollama"):
        say("✓ Ollama is installed.", "fg:green")
        ensure_ollama_current()
        return
    say("Ollama is the program that runs the AI models on your computer. It is not installed yet.", "fg:ansimagenta")
    cmd = install_command()
    if cmd is None:
        fail(f"Please install Ollama from https://ollama.com/download, then run {app_config.INSTALL_COMMAND} again.")
    say(f"Install command: {cmd[-1] if cmd[0] == 'sh' else ' '.join(cmd)}")
    if not ask(questionary.confirm("Install Ollama now?", default=True, qmark=QMARK, style=STYLE)):
        fail(f"Ollama is required. Install it from https://ollama.com/download, then run {app_config.INSTALL_COMMAND} again.")
    if not _succeeds(cmd):
        fail(f"The Ollama install did not finish. Install it from https://ollama.com/download, then run {app_config.INSTALL_COMMAND} again.")
    if shutil.which("ollama"):
        say("✓ Ollama installed.", "fg:green")
    else:
        say("✓ Ollama installed. Open a new terminal window afterwards so the `ollama` command works.", "fg:green")


# --- Step 4: models -----------------------------------------------------------

def _clean_model_name(raw: str) -> str:
    """Accept pasted 'VLM_1=name:tag' lines as well as plain names; drop markdown decoration."""
    return raw.strip().split("=", 1)[-1].strip().strip("`'\"*").strip()


def _validate_model(value: str):
    name = _clean_model_name(value)
    if not name:
        return "Please paste a model name, for example: name:tag"
    if name.upper() == "NONE":
        return "NONE means that option does not exist. Paste one of the other options."
    if not MODEL_NAME_PATTERN.match(name):
        return "That does not look like a model name. Paste only the name, for example: name:tag"
    return True


def _ask_name(label: str, default: str = "") -> str:
    """Ask for a model name. Returns '' when the box is left empty (clears the field)."""
    def validate(value: str):
        return True if not value.strip() else _validate_model(value)
    return _clean_model_name(ask(questionary.text(
        label, validate=validate, default=default, qmark=QMARK, style=STYLE,
        instruction="(paste it here, leave empty to clear) ")))


def parse_model_set(raw: str):
    """(vlm, embedding, text_model) from one pasted set, or None unless it holds exactly three model names.

    The AI is asked for ``vision-name:tag,text-name:tag,embedding-name:tag`` on one line, in that order. Commas, semicolons,
    spaces and line breaks all separate the names, and backticks, quotes and brackets around them are ignored."""
    names = [name for name in re.split(r"[,;|\s]+", re.sub(r"[`'\"*()\[\]]", "", raw)) if name]
    if len(names) != 3 or any(_validate_model(name) is not True for name in names):
        return None
    vlm, text_model, embedding = names
    return vlm, embedding, text_model


def _discard_pending_input() -> None:
    """Drop the keys the prompt read ahead but did not use, such as the rest of a multi-line paste (a terminal without
    bracketed paste submits only the first line). prompt_toolkit would otherwise hand them to the next menu."""
    try:
        from prompt_toolkit.input import create_input
        from prompt_toolkit.input.typeahead import clear_typeahead
        clear_typeahead(create_input())
    except Exception:  # best effort: if it cannot be cleared the leftovers are simply replayed
        pass


def _paste_set():
    """Ask for one set from the AI's answer. Returns (vlm, embedding, text_model), or None when it is left empty or unreadable."""
    pasted = ask(questionary.text(
        "Paste one set:", qmark=QMARK, style=STYLE, instruction="(leave empty to go back) "))
    if "\n" not in pasted:
        _erase_last_line()
    if not pasted.strip():
        return None
    names = parse_model_set(pasted)
    if names is None:
        _discard_pending_input()
        say("✗ Parsing failed. Please try again, or write the model names manually.", RED)
    return names


def _erase_last_line() -> None:
    """Remove the answered prompt line so the next menu redraws in the same place."""
    print("\033[1A\033[2K", end="", flush=True)


def _field_choice(label: str, value: str, key: str):
    """Menu row for a model field: green once filled, light red while unset."""
    if value:
        return questionary.Choice([("fg:ansigreen", f"○ {label} {value}")], value=key)
    return questionary.Choice([("fg:ansibrightred", f"○ {label} not set")], value=key)


def ask_models(vlm: str = "", embedding: str = "", text_model: str = ""):
    """One menu: open any field, copy the AI prompt (which adds a row on top to paste its answer), or continue once all names are set.

    Returns (vlm, embedding, text_model)."""
    say("Fill in all three model names, then choose Continue.\n")
    copied = asked = pasted = False
    while True:
        ready = bool(vlm and embedding and text_model)
        choices = []
        if asked and not pasted:  # only once the prompt was copied, and until a set was read
            choices += [questionary.Choice(f"○ {PASTE_SET}", value="paste"), questionary.Separator()]
        choices += [
            _field_choice("VLM model:      ", vlm, "vlm"),
            _field_choice("Embedding model:", embedding, "embedding"),
            _field_choice("Text model:     ", text_model, "text"),
            questionary.Separator(),
            questionary.Choice([("fg:ansigreen", f"○ {COPIED}")] if copied else f"○ {ASK_AI}", value="ask"),
            questionary.Separator(),
            questionary.Choice("○ Continue" if ready else "Continue", value="continue",
                               disabled=None if ready else "enter all three model names first"),
        ]
        default = "continue" if ready else "paste" if asked and not pasted else next(
            key for key, value in (("vlm", vlm), ("embedding", embedding), ("text", text_model)) if not value)
        choice = ask(questionary.select(
            "What would you like to do?", choices=choices, default=default,
            qmark=QMARK, pointer=POINTER, style=STYLE))
        if choice == "continue":
            return vlm, embedding, text_model
        _erase_last_line()
        if choice == "ask":
            copied = offer_prompt_copy()
            asked = True
        elif choice == "paste":
            names = _paste_set()
            if names:
                vlm, embedding, text_model = names
                pasted = True
        elif choice == "vlm":
            vlm = _ask_name("VLM model name:", vlm)
            _erase_last_line()
        elif choice == "embedding":
            embedding = _ask_name("Embedding model name:", embedding)
            _erase_last_line()
        else:
            text_model = _ask_name("Text model name:", text_model)
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


def review_models(vlm: str, embedding: str, text_model: str, local: bool):
    """Verify both models exist, show their specs, then ask Continue / Go back.

    ``local`` is whether Ollama runs on this computer; the disk-space check only makes sense then.
    Returns (proceed, names_not_found)."""
    say("\nChecking the models on ollama.com…")
    specs, problems, missing = {}, [], set()
    for name in dict.fromkeys([vlm, embedding, text_model]):
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
    for label, name in (("VLM model", vlm), ("Embedding model", embedding), ("Text model", text_model)):
        say(f"{label}: {name}", "bold")
        for key, value in specs[name]["rows"]:
            say(f"    {key + ':':<15}{value}")
        say("")
    total = sum(specs[name]["bytes"] for name in dict.fromkeys([vlm, embedding, text_model]))
    enough_space = True
    if local:
        # Ollama keeps its models in its own folder, usually on the home drive; this is an estimate.
        free = shutil.disk_usage(Path.home()).free
        say(f"Total download: {_fmt_bytes(total)}    Free disk space (home drive, estimate): {_fmt_bytes(free)}\n", "bold")
        enough_space = total * 1.1 <= free
        if not enough_space:
            say("✗ Not enough free disk space for these models. Free up space or go back and choose smaller models.\n", RED)
    else:
        say(f"Total download: {_fmt_bytes(total)}    (disk space on the Ollama computer is not checked)\n", "bold")
    say("Once Imagent is set up, these models cannot be changed.\n", "fg:ansimagenta")

    choice = ask(questionary.select(
        "Ready to download?",
        choices=[
            questionary.Choice("○ ← Go back and change the models", value="back"),
            questionary.Separator(),
            questionary.Choice("○ Continue: download the models" if enough_space else "Continue: download the models", value="go",
                               disabled=None if enough_space else "not enough disk space"),
        ],
        default="go" if enough_space else "back",
        qmark=QMARK, pointer=POINTER, style=STYLE,
    ))
    return choice == "go", missing


# --- Step 5: download ---------------------------------------------------------

@contextmanager
def ollama_client(host: str):
    """Yield a connected Ollama client, starting `ollama serve` only if needed and stopping it afterwards.

    Yields None (after printing the reason) when Ollama cannot be reached or started."""
    # needs the packages that ensure_packages installed
    from vlm.client import MIN_OLLAMA_VERSION, is_local, is_reachable, make_client, server_version, start_server, stop_server
    client = make_client(host)

    started = None
    try:
        if not is_reachable(client):
            if not is_local(host):
                say(f"✗ Cannot reach Ollama at {host}.", RED)
                yield None
                return
            if not shutil.which("ollama"):
                say(f"✗ The `ollama` command was not found. Open a new terminal window and run {app_config.INSTALL_COMMAND} again.", RED)
                yield None
                return
            say("Starting Ollama in the background…")
            started = start_server(client, host)
            if started is None:
                say("✗ Ollama did not start. Open the Ollama app (or run `ollama serve`) and try again.", RED)
                yield None
                return
        version = server_version(host)
        if version is not None and version < MIN_OLLAMA_VERSION:
            say(f"✗ The Ollama at {host} is version {'.'.join(map(str, version))}, but Imagent needs "
                f"{'.'.join(map(str, MIN_OLLAMA_VERSION))} or newer. Update it (on that computer, if it is another one), "
                f"then run {app_config.INSTALL_COMMAND} again.", RED)
            yield None
            return
        yield client
    finally:
        if started is not None:
            stop_server(started)
            say("Stopped the temporary Ollama server.")


def _normalize(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


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


def _remember_pulled(name: str) -> None:
    """Note in install_record.json that Imagent downloaded this model, so uninstalling removes it (and only such models)."""
    record = app_config.load_install_record()
    models = [m for m in record.get("ollama_models", []) if isinstance(m, str)]
    if _normalize(name) not in models:
        models.append(_normalize(name))
    app_config.save_install_record({**record, "ollama_models": models})


def _pull_models(client, models: list) -> None:
    """Pull each model that is not installed yet, showing progress."""
    installed = {_normalize(m.model) for m in client.list().models}
    for name in dict.fromkeys(models):
        if _normalize(name) in installed:
            say(f"✓ {name} is already downloaded.", "fg:ansigreen")
            continue
        say(f"Downloading {name}…")
        _pull_with_progress(client, name)
        _remember_pulled(name)
        say(f"✓ {name} downloaded.", "fg:ansigreen")


def verify_models(client, vlm: str, embedding: str, text_model: str):
    """Check that the downloaded models can do their job, before they are locked in.

    Returns (settings, problems). ``problems`` maps a model name to what is wrong with it."""
    problems = {}
    for name, needed, what in ((vlm, "vision", "a vision model"), (embedding, "embedding", "an embedding model"),
                               (text_model, "completion", "a text chat model")):
        try:
            capabilities = client.show(name).capabilities
        except Exception as exc:
            problems[name] = f"could not be inspected ({type(exc).__name__}: {exc})."
            continue
        if capabilities is None:
            problems[name] = f"cannot be checked because your Ollama is too old. Update Ollama, then run {app_config.INSTALL_COMMAND} again."
        elif needed not in capabilities:
            problems[name] = f"is not {what}. Ollama lists its abilities as: {', '.join(capabilities) or 'none'}."
    if problems:
        return None, problems
    try:
        dimension = len(client.embed(model=embedding, input=["Imagent setup check"],
                                     options=app_config.EMBED_OPTIONS).embeddings[0])
    except Exception as exc:
        return None, {embedding: f"did not produce an embedding ({type(exc).__name__}: {exc})."}
    if dimension == 0:
        return None, {embedding: "returned an empty embedding."}
    digests = {m.model: m.digest for m in client.list().models}
    return {
        "vlm_model": _normalize(vlm), "embedding_model": _normalize(embedding), "judge_model": _normalize(text_model),
        "vlm_digest": digests.get(_normalize(vlm)), "embedding_digest": digests.get(_normalize(embedding)),
        "judge_digest": digests.get(_normalize(text_model)),
        "embedding_dim": dimension,
    }, {}


def download_and_verify(host: str, vlm: str, embedding: str, text_model: str):
    """Download all models and verify them. Returns (settings, names_to_clear); settings is None on failure."""
    with ollama_client(host) as client:
        if client is None:
            return None, set()
        try:
            _pull_models(client, [vlm, embedding, text_model])
        except Exception as exc:
            print()
            say(f"✗ Download failed: {exc}", RED)
            return None, set()
        say("Checking that the models work…")
        try:
            settings, problems = verify_models(client, vlm, embedding, text_model)
        except Exception as exc:
            say(f"✗ Could not check the models: {exc}", RED)
            return None, set()
    for name, reason in problems.items():
        say(f"✗ '{name}' {reason}", RED)
    return settings, set(problems)


# --- Choosing models (first run only) ------------------------------------------

def choose_models(saved_host: str):
    """Input boxes -> verify on ollama.com -> specs -> download -> verify.

    Returns (settings, host). The caller locks the settings into the database."""
    from vlm.client import is_local  # needs the packages that ensure_packages installed
    step("Models")
    host = saved_host or DEFAULT_HOST
    vlm = embedding = text_model = ""
    while True:
        vlm, embedding, text_model = ask_models(vlm, embedding, text_model)
        proceed, missing = review_models(vlm, embedding, text_model, is_local(host))
        if not proceed:
            # Clear only the names that were not found; keep the valid ones.
            vlm = "" if vlm in missing else vlm
            embedding = "" if embedding in missing else embedding
            text_model = "" if text_model in missing else text_model
            step("Models")
            continue
        step("Download")
        settings, rejected = download_and_verify(host, vlm, embedding, text_model)
        if settings is not None:
            return settings, host
        vlm = "" if vlm in rejected else vlm
        embedding = "" if embedding in rejected else embedding
        text_model = "" if text_model in rejected else text_model
        step("Models")


# --- Re-run: status of the locked models ---------------------------------------

def _installed(client) -> dict:
    return {_normalize(m.model): (m.size or 0, m.digest) for m in client.list().models}


def _reinstall(client, name: str) -> None:
    """Pull a locked model that was removed from Ollama, after a disk-space check."""
    try:
        needed = fetch_model_specs(name)["bytes"]
        if needed * 1.1 > shutil.disk_usage(Path.home()).free:
            say(f"✗ Not enough free disk space for {name} ({_fmt_bytes(needed)}). Free up space and try again.", RED)
            return
    except ModelNotFound:
        say(f"✗ '{name}' no longer exists on ollama.com, so it cannot be reinstalled.", RED)
        return
    except RegistryUnreachable:
        pass  # offline check failed; let the pull report any real problem
    try:
        _pull_models(client, [name])
    except Exception as exc:
        print()
        say(f"✗ Download failed: {exc}", RED)


def manage_models(settings: dict, host: str) -> None:
    """Setup was run before: show the locked models and offer to reinstall one that was removed."""
    say("Setup was already completed. Checking your installed models…\n")
    with ollama_client(host) as client:
        if client is None:
            sys.exit(1)
        while True:
            try:
                installed = _installed(client)
            except Exception as exc:
                fail(f"Could not read the model list from Ollama: {exc}")

            # One entry per model name; the text model may be the same model as the VLM.
            locked: dict = {}
            for role, key in (("VLM", "vlm"), ("Embedding", "embedding"), ("Text", "judge")):
                entry = locked.setdefault(settings[f"{key}_model"], {"roles": [], "digest": settings[f"{key}_digest"]})
                entry["roles"].append(role)
            in_use = [(" + ".join(entry["roles"]), name, entry["digest"]) for name, entry in locked.items() if name in installed]
            missing = [(" + ".join(entry["roles"]), name) for name, entry in locked.items() if name not in installed]
            others = sorted(n for n in installed if n not in locked)
            width = max([len(n) for n in [*(name for _, name, _ in in_use), *others]] + [1])

            say("Installed models:", "bold")
            for role, name, digest in in_use:
                say(f"✓ {name:<{width}}  {_fmt_bytes(installed[name][0]):>8}  {role} (in use)", "fg:ansigreen")
                if digest and installed[name][1] != digest:
                    say("    ⚠ This is not the same version it was when you set up Imagent.", "fg:ansimagenta")
            for name in others:
                say(f"  {name:<{width}}  {_fmt_bytes(installed[name][0]):>8}")
            if not installed:
                say("  (no models installed)")
            say("")
            say("Note: the models cannot be changed once set. Each embedding model produces different",
                "fg:ansibrightblack")
            say("vectors for the same text, so switching models would break all existing image searches.",
                "fg:ansibrightblack")
            say("")

            choices = [
                questionary.Choice(
                    [(RED, f"○ {role} model: {name} is not installed. Select to reinstall")], value=("reinstall", name))
                for role, name in missing
            ]
            if choices:
                choices.append(questionary.Separator())
            choices.append(questionary.Choice("○ Done", value=("done", "")))
            action, name = ask(questionary.select(
                "What would you like to do?", choices=choices, default=choices[0].value if missing else ("done", ""),
                qmark=QMARK, pointer=POINTER, style=STYLE))
            _erase_last_line()

            if action == "done":
                say("✓ Setup complete.", "fg:green")
                return
            _reinstall(client, name)
            say("")


# --- Main ---------------------------------------------------------------------

def main() -> None:
    make_output_safe()
    if not sys.stdin.isatty():
        fail(f"Setup needs an interactive terminal. Run it directly: {app_config.INSTALL_COMMAND}")

    if RELAUNCH_FLAG not in sys.argv:
        step("Your computer")
        say(system_details())
        step("Required packages")
    ensure_packages()

    step("Text reading")
    prepare_ocr()

    step("Ollama")
    ensure_ollama_installed()

    try:
        conn = init_db(str(app_config.DB_PATH))
    except RuntimeError as exc:  # SQLite too old for the schema
        fail(str(exc))
    try:
        settings = get_model_settings(conn)
    finally:
        conn.close()
    host = app_config.load_config().get("ollama_host", "")

    if settings is not None:
        step("Installed models")
        if not host:  # config.json was deleted; the models are still locked in the database
            host = DEFAULT_HOST
            app_config.save_config({"ollama_host": host})
        manage_models(settings, host)
        return

    settings, host = choose_models(host)
    app_config.save_config({"ollama_host": host})
    conn = init_db(str(app_config.DB_PATH))
    try:
        with conn:
            set_model_settings(conn, settings)  # last step: this row marks setup as complete
    except sqlite3.IntegrityError:
        fail("Imagent was already set up by another run of setup.")
    finally:
        conn.close()
    say(f"\n✓ Setup complete. Your models are locked in {app_config.DB_PATH}", "fg:green")


if __name__ == "__main__":
    run_cli(main, cancelled="Setup cancelled. Run it again any time.")
