#!/usr/bin/env python3
"""One-command installer and first-run setup: show system details, create .venv and check its
packages, check Python and Ollama versions, collect and verify the model names, then lock them in the
Imagent database. Missing or outdated packages are installed automatically (only into .venv); Ollama
and a broken .venv are only fixed after you agree. The models cannot be changed afterwards; running this again only
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
from db.operations import get_device_settings, get_model_settings, set_device_settings, set_model_settings
from pipeline.console import make_output_safe, run_cli
from pipeline.progress import SearchProgress

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
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


def _metal_gpu_memory_gb():
    """Memory the macOS GPU may use in GB, as Metal reports it (recommendedMaxWorkingSetSize); None if Metal cannot be asked."""
    if platform.system() != "Darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.objc_msgSend.restype = ctypes.c_uint64
        device = metal.MTLCreateSystemDefaultDevice()
        if not device:
            return None
        size = objc.objc_msgSend(device, objc.sel_registerName(b"recommendedMaxWorkingSetSize"))
        return size / 2**30 if size else None
    except (OSError, AttributeError, TypeError):
        return None


def system_details() -> str:
    free_disk = shutil.disk_usage(Path.home()).free / 2**30
    lines = [
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- CPU: {_cpu()}",
        f"- RAM: {_ram_gb()}",
        f"- GPU: {_gpu()}",
    ]
    metal_gb = _metal_gpu_memory_gb()
    if metal_gb is not None:
        lines.append(f"- GPU memory usable (reported by Metal for this computer): {metal_gb:.1f} GB")
    lines.append(f"- Free disk space: {free_disk:.0f} GB")
    return "\n".join(lines)


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
6. Memory: model weights are fully loaded (no memory-mapping). Every set must fit entirely on the GPU, with no CPU offload and no swapping, in about 85% of my VRAM (or RAM if there is no dedicated GPU). With shared (unified) memory the GPU cannot use all of the RAM: if "GPU memory usable" is listed above, it is the figure to use, so take 85% of that and do not look it up. Check both cases; the vision and text models are never loaded together:
   - indexing: the vision model AND the embedding model are loaded at the same time, so both must fit together. Count the vision model's weights, its KV cache at a context of {app_config.OLLAMA_NUM_CTX} tokens and its image-encoder working memory for images up to {app_config.IMAGE_MAX_LONG_EDGE} px on the long edge (reserved for a worst-case image), plus the embedding model's weights and its own KV cache and compute buffers at the context Ollama gives it by default; those buffers can make an embedding model several times larger in memory than its weight file. Ollama does not publish buffer sizes, so estimate them and leave a safety margin instead of declining a set for lack of an exact figure;
   - searching: the text model AND the embedding model are loaded at the same time. Count the text model's weights and its KV cache at a context of {app_config.OLLAMA_NUM_CTX} tokens, plus the embedding model in memory as above.
7. Do the memory sums yourself and do not show the working.

## Answer
Give three sets. A set is one complete choice: a vision model, a text model and an embedding model that work together. Set 1 is the best balance and the one you recommend; set 2 is a faster or lighter alternative; set 3 is a more accurate one that still fits. Only give a set if you verified the tag and capabilities of all three of its models (memory is estimated, rule 6); if there are fewer than three, give fewer and say so.

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
    """Install requirements.txt with a live line (spinner and time so far). pip's own output is held back so it cannot
    break that line, and shown only if pip fails."""
    cmd = [python, "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)]
    progress = SearchProgress()
    progress.step("Installing the required packages")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=INSTALL_TIMEOUT)
        installed, details = result.returncode == 0, (result.stderr or result.stdout).strip()
    except subprocess.TimeoutExpired:
        installed, details = False, "pip did not finish in time."
    if installed:
        progress.close()
        return
    progress.fail("failed")
    progress.close()
    if details:
        print(details, file=sys.stderr)
    fail("Could not install the required packages (the messages above say which one). Check your internet connection and "
         f"run {app_config.INSTALL_COMMAND} again. If a package has no version for your Python ({platform.python_version()}), try another Python, "
         "such as 3.11 or 3.12.")


def _sync_packages(python: str) -> None:
    """Install into .venv what requirements.txt needs and it lacks or has too old, without asking. Nothing outside .venv changes."""
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
    say("Installing them now (this can take a minute). They go into .venv only; Python and the packages elsewhere on your computer are not touched.")
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
    """Load the OCR engine now. Its models come with the package, so nothing is downloaded, and Imagent never needs the internet once set up."""
    say("Loading the text-reading (OCR) engine…")
    try:
        from image_processing.ocr import check_available
        check_available()
    except Exception as exc:
        fail(f"The text-reading (OCR) engine could not be loaded ({type(exc).__name__}: {exc}). "
             f"Run {app_config.INSTALL_COMMAND} again; if it keeps failing, delete the .venv folder first so the packages are installed afresh.")
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


def parse_model_set(raw: str):
    """(vlm, embedding, text_model) from one pasted set, or None unless it holds exactly three model names.

    The AI is asked for ``vision-name:tag,text-name:tag,embedding-name:tag`` on one line, in that order. Commas, semicolons,
    spaces and line breaks all separate the names, and backticks, quotes and brackets around them are ignored."""
    names = [name for name in re.split(r"[,;|\s]+", re.sub(r"[`'\"*()\[\]]", "", raw)) if name]
    if len(names) != 3 or any(_validate_model(name) is not True for name in names):
        return None
    vlm, text_model, embedding = names
    return vlm, embedding, text_model


def _erase_last_line() -> None:
    """Remove the answered prompt line so the next menu redraws in the same place."""
    print("\033[1A\033[2K", end="", flush=True)


def ask_models(vlm: str = "", embedding: str = "", text_model: str = ""):
    """One screen of editable rows: type or paste each model name where it is shown, copy the AI prompt (which adds a row to
    paste its answer into), and choose Continue once all three names are set. Nothing is cleared while you type.

    Returns (vlm, embedding, text_model)."""
    from prompt_toolkit import Application
    from prompt_toolkit.application import get_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.filters import Condition, has_focus, is_done
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import BufferControl, ConditionalContainer, FormattedTextControl, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor

    say("Fill in all three model names, then choose Continue.\n")
    state = {"copied": False, "asked": False, "pasted": False, "message": ""}

    def one_line(buffer):  # a pasted line break would show as ^J
        if "\n" in buffer.text or "\r" in buffer.text:
            buffer.document = Document(" ".join(buffer.text.split()))

    def make_buffer(text=""):
        return Buffer(document=Document(text, len(text)), multiline=False, on_text_changed=one_line)

    fields = {"vlm": make_buffer(vlm), "embedding": make_buffer(embedding), "text": make_buffer(text_model)}
    paste = make_buffer()
    pointer = lambda window: POINTER + " " if not is_done() and get_app().layout.has_focus(window) else " " * (len(POINTER) + 1)

    def ready():
        return all(buffer.text.strip() for buffer in fields.values())

    show_paste = Condition(lambda: state["asked"] and not state["pasted"]) & ~is_done
    show_message = Condition(lambda: bool(state["message"])) & ~is_done

    def blank(visible=~is_done):
        return ConditionalContainer(Window(height=1), filter=visible)

    def text_row(label, buffer, filled_green=True):
        window = Window(
            BufferControl(buffer, input_processors=[ConditionalProcessor(
                AfterInput("not set", style="fg:ansibrightred"), Condition(lambda: filled_green and not buffer.text))]),
            height=1, style=lambda: "fg:ansigreen" if filled_green and buffer.text else "")
        style = lambda: "fg:ansigreen" if buffer.text or not filled_green else "fg:ansibrightred"
        title = Window(FormattedTextControl(lambda: [(style(), f"{pointer(window)}○ {label} ")]), height=1, dont_extend_width=True)
        return window, VSplit([title, window])

    rows = {}
    field_rows = []
    for key, label in (("vlm", "VLM model:      "), ("embedding", "Embedding model:"), ("text", "Text model:     ")):
        rows[key], container = text_row(label, fields[key])
        field_rows.append(container)
    rows["paste"], paste_row = text_row("Paste one set:", paste, filled_green=False)

    def button(text, style):
        window = Window(FormattedTextControl(lambda: [(style(), f"{pointer(window)}○ {text()}")], focusable=True), wrap_lines=True)
        return window

    rows["ask"] = button(lambda: COPIED if state["copied"] else ASK_AI, lambda: "fg:ansigreen" if state["copied"] else "")
    rows["continue"] = button(lambda: "Continue" if ready() else "Continue (enter all three model names first)",
                              lambda: "" if ready() else "fg:ansibrightblack italic")
    visible = {"vlm": Condition(lambda: True), "embedding": Condition(lambda: True), "text": Condition(lambda: True),
               "paste": show_paste, "ask": ~is_done, "continue": ~is_done}

    def initial_focus():
        if ready():
            return "continue"
        if state["asked"] and not state["pasted"]:
            return "paste"
        return next(key for key, buffer in fields.items() if not buffer.text.strip())

    def order():
        return [rows[key] for key in ("vlm", "embedding", "text", "paste", "ask", "continue") if visible[key]()]

    def commit(app):
        """Tidy the name in the row being left; False (and say why) when it is not a model name."""
        for key, buffer in fields.items():
            if app.layout.has_focus(rows[key]):
                name = _clean_model_name(buffer.text)
                problem = _validate_model(name) if name else True
                if problem is not True:
                    state["message"] = "✗ " + problem
                    return False
                buffer.document = Document(name)
        state["message"] = ""
        return True

    def move(app, step):
        if not commit(app):
            return
        windows = order()
        current = app.layout.current_window
        index = windows.index(current) if current in windows else 0
        app.layout.focus(windows[(index + step) % len(windows)])

    bindings = KeyBindings()
    for key, step in (("up", -1), ("s-tab", -1), ("down", 1), ("tab", 1)):
        bindings.add(key)(lambda event, step=step: move(event.app, step))
    for key in fields:
        bindings.add("enter", filter=has_focus(rows[key]))(lambda event: move(event.app, 1))

    @bindings.add("enter", filter=has_focus(rows["paste"]))
    def _(event):
        names = parse_model_set(paste.text)
        if names is None:
            state["message"] = "✗ Parsing failed. Please try again, or write the model names manually." if paste.text.strip() else ""
            paste.document = Document()
            return
        for key, name in zip(("vlm", "embedding", "text"), names):
            fields[key].document = Document(name)
        event.app.layout.focus(rows["continue"])
        state["pasted"], state["message"] = True, ""

    @bindings.add("enter", filter=has_focus(rows["ask"]))
    def _(event):
        state["asked"], state["message"] = True, ""
        if copy_to_clipboard(build_ai_prompt()):
            state["copied"] = True
            if not state["pasted"]:
                event.app.layout.focus(rows["paste"])
        else:  # nowhere to copy it to: leave the screen, print the prompt for copying by hand, and come back
            event.app.erase_when_done = True
            event.app.exit(result="print")

    @bindings.add("enter", filter=has_focus(rows["continue"]))
    def _(event):
        if ready():
            event.app.exit(result="continue")
        else:
            state["message"] = "✗ Enter all three model names first."

    @bindings.add("c-c")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt)

    while True:
        layout = HSplit([
            ConditionalContainer(HSplit([paste_row, Window(height=1, content=FormattedTextControl([("fg:ansibrightblack", INDENT + "  " + PASTE_SET)])), blank()]),
                                 filter=show_paste),
            *field_rows, blank(),
            ConditionalContainer(HSplit([rows["ask"], blank(), rows["continue"]]), filter=~is_done),
            ConditionalContainer(Window(FormattedTextControl(lambda: [(RED, INDENT + state["message"])]), height=1), filter=show_message),
        ])
        app = Application(layout=Layout(layout, focused_element=rows[initial_focus()]), key_bindings=bindings, erase_when_done=False)
        try:
            result = app.run()
        except (KeyboardInterrupt, EOFError):
            sys.exit("\nSetup cancelled. Run it again any time.")
        if result == "continue":
            return tuple(fields[key].text.strip() for key in ("vlm", "embedding", "text"))
        state["copied"] = offer_prompt_copy()  # "print": offer_prompt_copy tries the clipboard once more, then prints the prompt


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


def measure_computer(client, vlm_model: str, embedding_model: str):
    """Measure this computer with the vision model and the embedding model beside it (vlm/calibration.py), with no questions asked.

    Returns (device settings, None), or (None, what is wrong with the model) when it does not run on this computer."""
    from vlm.calibration import NotPossible, calibrate
    say("Measuring this computer with the vision model. This runs by itself and can take a few minutes…")
    try:
        device = calibrate(client, vlm_model, embedding_model, lambda message: say(f"  {message}"))
    except NotPossible as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"could not be measured ({type(exc).__name__}: {exc})."
    say(f"✓ Pictures up to {device['image_long_edge']} pixels, a memory window of {device['num_ctx']}, "
        f"waiting up to {device['timeout_seconds']} seconds for a reply.", "fg:ansigreen")
    if device["one_model_at_a_time"]:
        say("⚠ The vision and embedding models do not fit in memory together on this computer. Imagent will load one at a time, "
            "which makes indexing a little slower.", "fg:ansimagenta")
    return device, None


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
        from vlm.client import pin_model
        original = _normalize(embedding)
        try:
            embedding, _ = pin_model(client, original)
            app_config.record_pinned_model(embedding, original)
            say(f"✓ Saved a copy of {original} that stays exactly as it is now, even if the model is updated later.", "fg:ansigreen")
        except Exception as exc:
            embedding = original
            say(f"⚠ Could not save a fixed copy of {original} ({type(exc).__name__}: {exc}). Imagent will use it as it is "
                "and stop if it ever changes.", "fg:ansimagenta")
        say("Checking that the models work…")
        try:
            settings, problems = verify_models(client, vlm, embedding, text_model)
        except Exception as exc:
            say(f"✗ Could not check the models: {exc}", RED)
            return None, set()
        if embedding in problems and embedding != original:
            problems[original] = problems.pop(embedding)
        if settings is not None:
            settings["device"], reason = measure_computer(client, settings["vlm_model"], settings["embedding_model"])
            if reason:
                settings, problems = None, {vlm: reason}
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
    """Pull a locked model that was removed from Ollama, after a disk-space check.

    A pinned copy (see vlm.client.pin_model) is not on ollama.com: it is made again from the model it was copied from,
    if that is still the same version."""
    from vlm.client import is_pinned, pinned_name
    if is_pinned(name):
        base = (app_config.load_install_record().get("pinned_models") or {}).get(name, "")
        try:
            digest = {m.model: m.digest for m in client.list().models}.get(base)
            if digest and pinned_name(base, digest) == name:
                client.copy(base, name)
                say(f"✓ {name} was made again from {base}.", "fg:ansigreen")
                return
        except Exception as exc:
            say(f"✗ Could not make {name} again ({type(exc).__name__}: {exc}).", RED)
            return
        say(f"✗ {name} was removed from Ollama, and {base or 'the model it was copied from'} is not the same version any more, "
            f"so the original cannot be restored. Run {app_config.RUN_COMMAND} and choose \"Re-embed images\" to rebuild the index "
            "with the model that is installed.", RED)
        return
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
                return
            _reinstall(client, name)
            say("")


def check_test_pictures() -> None:
    """Unpack the test pictures into Imagent's data folder and run the steps of indexing that need no model over them.

    Nothing is asked. A problem here is reported but does not stop setup: it is a check, and Imagent works without it."""
    from image_processing import sample_pictures
    folder = app_config.IMAGENT_HOME / "test_pictures"
    say("Preparing the test pictures and checking that this computer can read them…")
    findings: list[str] = []
    try:
        sample_pictures.unpack(folder)
        expected = sample_pictures.make_extremes(folder, say=lambda message: say(f"  {message}", "fg:ansimagenta"))
        outcomes = sample_pictures.check_pictures(folder, expected, say=findings.append)
    except Exception as exc:
        say(f"⚠ The test pictures could not be prepared ({type(exc).__name__}: {exc}). Imagent works without them.", "fg:ansimagenta")
        return
    summary = ", ".join(f"{count} {outcome}" for outcome, count in sorted(outcomes.items()))
    say(f"✓ Checked {sum(outcomes.values())} test pictures: {summary}.", "fg:ansigreen")
    for finding in findings:
        say(f"⚠ {finding}", "fg:ansimagenta")


def measure_again_if_needed(settings: dict, host: str) -> None:
    """Setup was run before: measure this computer if it never was, or if the vision model is not the one it was measured with."""
    conn = init_db(str(app_config.DB_PATH))
    try:
        device = get_device_settings(conn)
    finally:
        conn.close()
    with ollama_client(host) as client:
        if client is None:
            sys.exit(1)
        try:
            digest = {m.model: m.digest for m in client.list().models}.get(settings["vlm_model"])
        except Exception as exc:
            fail(f"Could not read the model list from Ollama: {exc}")
        if digest is None or (device is not None and device["vlm_digest"] == digest and device["one_model_at_a_time"] is not None):
            return  # a missing model was offered for reinstall above; a measured one is still right
        device, reason = measure_computer(client, settings["vlm_model"], settings["embedding_model"])
    if reason:
        fail(f"'{settings['vlm_model']}' {reason}")
    conn = init_db(str(app_config.DB_PATH))
    try:
        with conn:
            set_device_settings(conn, device, digest)
    finally:
        conn.close()


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
    from vlm.client import is_local
    if host and not is_local(host):
        say(f"⚠ config.json points to an Ollama on another computer ({host}). Imagent only works with an Ollama on this "
            f"computer, so the address is set back to {DEFAULT_HOST}.", "fg:ansimagenta")
        host = DEFAULT_HOST
        app_config.save_config({**app_config.load_config(), "ollama_host": host})

    if settings is not None:
        step("Installed models")
        if not host:  # config.json was deleted; the models are still locked in the database
            host = DEFAULT_HOST
            app_config.save_config({"ollama_host": host})
        manage_models(settings, host)
        measure_again_if_needed(settings, host)
        check_test_pictures()
        say("✓ Setup complete.", "fg:green")
        return

    settings, host = choose_models(host)
    device = settings.pop("device")
    app_config.save_config({"ollama_host": host})
    conn = init_db(str(app_config.DB_PATH))
    try:
        with conn:
            set_device_settings(conn, device, settings["vlm_digest"])
            set_model_settings(conn, settings)  # last step: this row marks setup as complete
    except sqlite3.IntegrityError:
        fail("Imagent was already set up by another run of setup.")
    finally:
        conn.close()
    check_test_pictures()
    say(f"\n✓ Setup complete. Your models are locked in {app_config.DB_PATH}", "fg:green")


if __name__ == "__main__":
    run_cli(main, cancelled="Setup cancelled. Run it again any time.")
