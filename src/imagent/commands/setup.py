"""
imagent/commands/setup.py — First-run setup (``imagent setup``): show system details, load the text-reading engine, check
Ollama, collect and verify the model names, save a fixed copy of each model, then lock them in the Imagent database.
Ollama is only fixed after you agree. The models cannot be changed afterwards; running this again only shows their status and
reinstalls a missing one. Starts Ollama only temporarily, to download the models.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

import questionary

from imagent import config as app_config
from imagent.db.connection import init_db
from imagent.db.operations import (get_device_settings, get_keep_deleted, get_model_settings, set_device_settings, set_keep_deleted,
                                   set_model_settings)
from imagent.pipeline.console import make_output_safe

DEFAULT_HOST = "http://localhost:11434"
CANCELLED = "Setup cancelled. Run it again any time."
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9._-]+)?$")
REGISTRY = "https://registry.ollama.ai/v2"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
KEEP_DELETED_CHOICES = (100, 500, 1000, 2000)  # how many deleted pictures to remember, offered at the end of setup
KEEP_DELETED_DEFAULT = 1000
ENTER_A_NUMBER = "enter a number"
RED = "fg:ansibrightred"
INDENT = "  "  # left padding for everything under a step heading
QMARK = INDENT + "?"
POINTER = INDENT + "»"

ASK_AI = "Not sure? Ask AI. Copy the prompt and give it to your favorite LLM."
COPIED = "✓ Prompt copied to clipboard. Paste it into your favorite LLM. (Select again to copy once more)"
PASTE_SET = "Paste the AI's answer: one set of three models (vision, text, embedding)."

STYLE = questionary.Style([
    ("qmark", "fg:ansidefault bold"),
    ("question", "fg:ansidefault bold"),
    ("pointer", "fg:ansidefault bold"),
    ("highlighted", "fg:ansidefault bold underline"),
    ("selected", "noreverse fg:ansidefault bold underline"),
    ("answer", "fg:ansidefault bold"),
    ("instruction", "fg:ansibrightblack"),
    ("disabled", "fg:ansibrightblack italic"),
])


# --- Output helpers -----------------------------------------------------------

def say(message: str = "", style: str = "") -> None:
    message = "\n".join(INDENT + line if line else line for line in message.split("\n"))
    questionary.print(message, style=style)


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

def failed_models_section(failed: dict | None) -> str:
    """The part of the AI prompt about models that were downloaded and tried on this computer and did not work: each with the reason."""
    if not failed:
        return ""
    lines = "\n".join(f"- {name} {' '.join(reason.split())}" for name, reason in failed.items())
    return f"""
## Models that already failed on this computer
They were downloaded and tried here. Do not suggest any of them, and do not suggest a larger model of the same family: choose smaller ones, or another family. A model may have failed for lack of memory, for being too slow, or because it could not answer in the required JSON form.
{lines}
"""


def build_ai_prompt(failed: dict | None = None) -> str:
    """The prompt to paste into an AI chat. ``failed`` maps each model that was tried on this computer and did not work to the reason."""
    return f"""I need to choose AI models for a local image search tool that runs on my computer through Ollama. Choose the smallest and fastest models that are accurate enough for the tasks below, not the largest that fit: more memory is a limit, never a goal, and a computer with a lot of memory does not need big models for these tasks.

## My computer
{system_details()}
{failed_models_section(failed)}
## The three models
- Vision model: looks at each image once and writes a JSON description (scene, objects, people, colors and mood, tags). A separate OCR step reads text in images. The descriptions must be accurate, and I may index thousands of images, so speed matters too.
- Text model: fixes typos in my search query and writes a short description of the image I am looking for. It never sees images. One request per search. This is an easy task that a small model does well, and it can be the vision model itself.
- Embedding model: turns the stored descriptions (about 500 tokens each) and my search queries (possibly in several languages) into vectors. Used when indexing and when searching.

## Rules
1. Search the web now. Do not rely on your memory.
2. Only models from https://ollama.com/library, each with an exact tag that you confirmed on its page so that `ollama pull <name>:<tag>` works. Never guess or invent a tag.
3. Capabilities: the vision model needs "vision", the text model needs "completion", the embedding model needs "embedding" (context of at least 1024 tokens, strong multilingual retrieval). The vision and text models must follow a JSON schema reliably and must not be thinking-only (reasoning) models. By default the text model is the same model as the vision model: one download fewer, and it is already loaded after indexing. Give a separate text model only when it is clearly faster for a search (about twice as fast) and just as good at fixing typos and at not inventing details I did not ask for; do not give a larger text model than the vision model.
4. Quantization of 4-bit or better, never lower.
5. Speed: a vision reply within a few seconds per image (never more than about 30 seconds) and a text reply within about 10 seconds. Indexing thousands of images must take hours, not days, on my computer. If two models are within about 30% of each other in speed, choose the more accurate one. Choose the faster one only when it is clearly faster (about twice as fast) and only slightly less accurate. Judge by published benchmarks and the model's own page, not by size, popularity or release date; an older maintained model is fine if the balance is better.
   Enough is enough: a bigger model is worth choosing only when it is clearly more accurate at these tasks (describing images in the JSON format above, and fixing typos in short queries), not merely higher on general benchmarks. When a smaller model of the same family is within a few percent on the tasks, choose the smaller one. Do not choose a model because it is the largest that fits.
6. Memory: this is only an upper limit, not a target; a set that uses much less memory is better, not worse. Model weights are fully loaded (no memory-mapping). Every set must fit entirely on the GPU, with no CPU offload and no swapping, in about 85% of my VRAM (or RAM if there is no dedicated GPU). With shared (unified) memory the GPU cannot use all of the RAM: if "GPU memory usable" is listed above, it is the figure to use, so take 85% of that and do not look it up. Check both cases; the vision and text models are never loaded together:
   - indexing: the vision model AND the embedding model are loaded at the same time, so both must fit together. Count the vision model's weights, its KV cache at a context of {app_config.OLLAMA_NUM_CTX} tokens and its image-encoder working memory for images up to {app_config.IMAGE_MAX_LONG_EDGE} px on the long edge (reserved for a worst-case image), plus the embedding model's weights and its own KV cache and compute buffers at the context Ollama gives it by default; those buffers can make an embedding model several times larger in memory than its weight file. Ollama does not publish buffer sizes, so estimate them and leave a safety margin instead of declining a set for lack of an exact figure;
   - searching: the text model AND the embedding model are loaded at the same time. Count the text model's weights and its KV cache at a context of {app_config.OLLAMA_NUM_CTX} tokens, plus the embedding model in memory as above.
7. Do the memory sums yourself and do not show the working.

## Answer
Give three sets. A set is one complete choice: a vision model, a text model and an embedding model that work together. Set 1 is the smallest set that is accurate enough and the one you recommend; set 2 is a faster or lighter alternative; set 3 is a more accurate one that still fits, and the table must show how much accuracy it adds, so that I can judge whether it is worth the extra size and time. Only give a set if you verified the tag and capabilities of all three of its models (memory is estimated, rule 6); if there are fewer than three, give fewer and say so.

Reply with exactly these three parts, in this order, and nothing else:
1. A table with one row per set and the columns: Set, Vision model, Text model, Embedding model, Memory (indexing / searching), Speed, Accuracy (the scores you used).
2. One or two sentences per set on the trade-offs against the other two sets: what I gain and what I give up.
3. The heading "Copy one set", then one code block per set (plain, no language name). Each code block holds one line and nothing else: the three model names, in the order vision, text, embedding, separated by commas.
   - Exact shape: visionname:7b,textname:8b,embeddingname:335m (when the text model is the vision model, write its name twice: visionname:7b,visionname:7b,embeddingname:335m)
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


def offer_prompt_copy(failed: dict | None = None) -> bool:
    """Copy the AI prompt to the clipboard. If that fails, print it for manual copying."""
    prompt = build_ai_prompt(failed)
    if copy_to_clipboard(prompt):
        return True
    say("Could not reach your clipboard. Select and copy the text below by hand,", "fg:ansimagenta")
    say("then paste it into any AI chat:\n", "fg:ansimagenta")
    print(prompt)
    return False


# --- Helpers ------------------------------------------------------------------

def prepare_ocr() -> None:
    """Load the OCR engine now. Its models come with the package, so nothing is downloaded, and Imagent never needs the internet once set up."""
    say("Loading the text-reading (OCR) engine…")
    try:
        from imagent.image_processing.ocr import check_available
        check_available()
    except Exception as exc:
        fail(f"The text-reading (OCR) engine could not be loaded ({type(exc).__name__}: {exc}). "
             f"Run {app_config.SETUP_COMMAND} again; if it keeps failing, reinstall Imagent (pipx reinstall imagent).")
    say("✓ Text reading is ready.", "fg:green")


# --- Step 2: Ollama -----------------------------------------------------------

class OllamaInstallError(Exception):
    """Installing Ollama failed; the message says why."""


OLLAMA_DESCRIPTION = "Runs the AI models on your computer. Imagent installed its own copy, which only Imagent uses."


def _ollama_command_version(command: str = "ollama"):
    """Version of the `ollama` program at ``command`` as numbers, e.g. (0, 6, 4); None if it cannot be told."""
    try:
        out = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout + out.stderr
    match = re.search(r"client version is (\d+)\.(\d+)\.(\d+)", text) or re.search(r"version is (\d+)\.(\d+)\.(\d+)", text)
    version = tuple(int(part) for part in match.groups()) if match else None
    return version if version and any(version) else None  # a development build reports 0.0.0


def _remove(path: Path) -> None:
    """Delete a file or folder if it exists."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _download_file(url: str, target: Path) -> str:
    """Download ``url`` into ``target``, showing progress. Returns the SHA-256 of what arrived."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "imagent"})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        free = shutil.disk_usage(target.parent).free
        if total and free < total:
            raise OllamaInstallError(f"There is not enough free disk space: the download is {_fmt_bytes(total)} and {_fmt_bytes(free)} is free.")
        done, shown = 0, 0.0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if time.monotonic() - shown >= 0.25:
                shown = time.monotonic()
                share = f"  {done * 100 // total}%  ({_fmt_bytes(done)} of {_fmt_bytes(total)})" if total else f"  {_fmt_bytes(done)}"
                print(f"\r\033[K{INDENT}Downloading Ollama{share}", end="", flush=True)
    print("\r\033[K", end="", flush=True)
    return digest.hexdigest()


def _check_tar_member(destination: Path, member: tarfile.TarInfo) -> None:
    """Refuse an archive entry that would be written outside ``destination`` or is not a plain file, folder or link."""
    root = destination.resolve()

    def inside(path: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            return False

    target = destination / member.name
    ok = (member.isreg() or member.isdir() or member.issym() or member.islnk()) and inside(target)
    if ok and member.issym():
        ok = inside(target.parent / member.linkname)
    if ok and member.islnk():
        ok = inside(destination / member.linkname)
    if not ok:
        raise OllamaInstallError(f"The download holds an unsafe entry ({member.name}), so it was not unpacked.")


def _unpack(archive: Path, destination: Path) -> None:
    """Unpack Ollama's .zip, .tgz or .tar.zst into the folder ``destination``, over what an earlier archive put there."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)  # removes any ../ or absolute path from the names itself
        return
    options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    if archive.name.endswith(".tar.zst"):
        import zstandard
        try:
            with open(archive, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader, \
                    tarfile.open(fileobj=reader, mode="r|") as bundle:
                for member in bundle:
                    _check_tar_member(destination, member)
                    bundle.extract(member, destination, **options)
        except zstandard.ZstdError as exc:
            raise OllamaInstallError(f"The download is damaged ({exc}).") from exc
        return
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            _check_tar_member(destination, member)
            bundle.extract(member, destination, **options)


TEGRA_RELEASE = Path("/etc/nv_tegra_release")  # exists on NVIDIA Jetson boards; names the JetPack generation (R35, R36)
DRM_DIR = Path("/sys/class/drm")  # the graphics cards on Linux, each with the vendor id of its maker
AMD_VENDOR_ID = "0x1002"


def _nvidia_gpu_present() -> bool:
    return bool(shutil.which("nvidia-smi")) and "GPU" in _run(["nvidia-smi", "-L"])


def _amd_gpu_present() -> bool:
    if platform.system() == "Windows":
        names = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"]).lower()
        return "amd" in names or "radeon" in names
    try:
        return any(card.read_text().strip().lower() == AMD_VENDOR_ID for card in DRM_DIR.glob("card*/device/vendor"))
    except OSError:
        return False


def _ollama_extras() -> dict:
    """The extra Ollama files this computer's GPU needs on top of the main build, as {name in the release's file names: what it is for}.
    Ollama's own installer does the same: AMD GPUs and NVIDIA Jetson boards need an extra file, an NVIDIA GPU needs none (its driver
    is what must be installed), and a Mac needs none."""
    system = platform.system()
    if system == "Linux" and TEGRA_RELEASE.is_file():
        text = TEGRA_RELEASE.read_text(errors="replace")
        for generation, extra in (("R36", "jetpack6"), ("R35", "jetpack5")):
            if generation in text:
                return {extra: "an NVIDIA Jetson board"}
        return {}
    if system in ("Linux", "Windows") and not _nvidia_gpu_present() and _amd_gpu_present():  # with both, the NVIDIA one is used
        return {"rocm": "an AMD GPU"}
    return {}


def install_ollama() -> None:
    """Download Ollama's own build for this computer into Imagent's tools folder, plus the extra file its GPU needs (AMD, Jetson).
    It needs no administrator rights and no change to PATH, and uninstalling removes it by deleting the folder. Stops setup, with the
    reason, when it cannot be done."""
    manual = f"Install Ollama from https://ollama.com/download, then run {app_config.SETUP_COMMAND} again."
    if app_config.ollama_build_prefix() is None:
        fail(f"Imagent has no Ollama build for this computer ({platform.system()} on {platform.machine()}). {manual}")
    root = app_config.OLLAMA_DIR
    program, partial, manifest = root / "program", root / "program.partial", root / app_config.OLLAMA_MANIFEST
    archive = None
    say("Installing Ollama for this project.")
    swapped = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        _remove(partial)
        _remove(manifest)
        # The release's list of files and their sums is trusted only when it is the one pinned in config.py; the sums in it are
        # then what each file is checked against.
        if _download_file(app_config.ollama_release_url(app_config.OLLAMA_MANIFEST), manifest) != app_config.OLLAMA_MANIFEST_SHA256:
            raise OllamaInstallError("The list of Ollama's files is not the one Imagent was made for (its checksum is different).")
        extras = _ollama_extras()
        downloads = app_config.ollama_downloads(manifest.read_text(encoding="utf-8", errors="replace"), tuple(extras))
        if not downloads:
            raise OllamaInstallError(f"Ollama {app_config.OLLAMA_VERSION} has no build for this computer ({platform.system()} on {platform.machine()}).")
        files = {file_name for _, file_name, _ in downloads}
        for extra, purpose in extras.items():
            stem = f"{app_config.ollama_build_prefix()}-{extra}"
            if any(stem + ending in files for ending in app_config.OLLAMA_ARCHIVE_ENDINGS):
                say(f"This computer has {purpose}, so the extra files for it are installed too.")
            else:
                say(f"This computer has {purpose}, but Ollama {app_config.OLLAMA_VERSION} has no extra files for it, so it may not be used.",
                    "fg:ansimagenta")
        for url, file_name, expected in downloads:
            archive = root / file_name
            _remove(archive)
            if _download_file(url, archive) != expected:
                raise OllamaInstallError("The download is not the file Ollama published (its checksum is different).")
            say("Unpacking…")
            _unpack(archive, partial)  # the extra files go over the build, into the same folder
            _remove(archive)
        _remove(program)
        os.replace(partial, program)  # a half-unpacked folder never looks installed
        swapped = True
        binary = app_config.private_ollama_binary()
        if binary is None:
            raise OllamaInstallError("The download does not contain the `ollama` program.")
        if subprocess.run([binary, "--version"], capture_output=True, timeout=60).returncode != 0:
            raise OllamaInstallError("The downloaded Ollama does not run on this computer.")
        (root / "models").mkdir(exist_ok=True)
        app_config.save_tool_info(root, "Ollama", OLLAMA_DESCRIPTION, app_config.OLLAMA_VERSION)
    except OllamaInstallError as exc:
        reason = str(exc)
    except OSError as exc:
        reason = "There is not enough free disk space." if exc.errno == errno.ENOSPC else f"{type(exc).__name__}: {exc}"
    except (tarfile.TarError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = None
    finally:
        if archive is not None:
            _remove(archive)
        _remove(manifest)
        _remove(partial)
    if reason is not None:
        if swapped:
            _remove(program)
        fail(f"The Ollama install did not finish. {reason} {manual}")
    say("✓ Ollama installed.", "fg:green")


def ensure_ollama_installed() -> None:
    """Use the Ollama that is here: Imagent's own copy, or the one on PATH if it is new enough. Otherwise install Imagent's own copy,
    and leave an older Ollama of the user's as it is."""
    from imagent.vlm.client import MIN_OLLAMA_VERSION  # needs the packages that ensure_packages installed
    if app_config.private_ollama_binary():
        say("✓ Ollama is installed (the copy Imagent installed).", "fg:green")
        return
    system_copy = shutil.which("ollama")
    if system_copy:
        version = _ollama_command_version(system_copy)
        if version is None or version >= MIN_OLLAMA_VERSION:
            say("✓ Ollama is installed.", "fg:green")
            return
        say(f"Your Ollama is version {'.'.join(map(str, version))}, but Imagent needs {'.'.join(map(str, MIN_OLLAMA_VERSION))} or newer. "
            "Imagent installs its own copy and leaves yours as it is.", "fg:ansimagenta")
    else:
        say("Ollama is the program that runs the AI models on your computer. It is not installed yet.", "fg:ansimagenta")
    install_ollama()


# --- Step 3: models -----------------------------------------------------------

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


def ask_models(vlm: str = "", embedding: str = "", text_model: str = "", failed: dict | None = None):
    """One screen of editable rows: type or paste each model name where it is shown, copy the AI prompt (which adds a row to
    paste its answer into; a good paste continues at once), and choose Continue once all three names are set. Nothing is cleared while you type.

    ``failed`` maps the models that were tried on this computer and did not work to the reason; the prompt tells the AI about them.

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
    state = {"copied": False, "asked": False, "message": ""}

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

    show_paste = Condition(lambda: state["asked"]) & ~is_done
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
        # dont_extend_height: without it the row stretches to fill the free height below the cursor, leaving a gap of blank lines
        window = Window(FormattedTextControl(lambda: [(style(), f"{pointer(window)}○ {text()}")], focusable=True), wrap_lines=True,
                        dont_extend_height=True)
        return window

    rows["ask"] = button(lambda: COPIED if state["copied"] else ASK_AI, lambda: "fg:ansigreen" if state["copied"] else "")
    rows["continue"] = button(lambda: "Continue" if ready() else "Continue (enter all three model names first)",
                              lambda: "" if ready() else "fg:ansibrightblack italic")
    visible = {"vlm": Condition(lambda: True), "embedding": Condition(lambda: True), "text": Condition(lambda: True),
               "paste": show_paste, "ask": ~is_done, "continue": ~is_done}

    def initial_focus():
        if ready():
            return "continue"
        if state["asked"]:
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
        event.app.exit(result="continue")  # a good paste is the answer: no second Continue

    @bindings.add("enter", filter=has_focus(rows["ask"]))
    def _(event):
        state["asked"], state["message"] = True, ""
        if copy_to_clipboard(build_ai_prompt(failed)):
            state["copied"] = True
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
        state["copied"] = offer_prompt_copy(failed)  # "print": offer_prompt_copy tries the clipboard once more, then prints the prompt


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
            questionary.Choice("○ ✓ Continue: download the models" if enough_space else "Continue: download the models", value="go",
                               disabled=None if enough_space else "not enough disk space"),
            questionary.Choice("○ ← Go back and change the models", value="back"),
        ],
        qmark=QMARK, pointer=POINTER, style=STYLE,
    ))
    return choice == "go", missing


# --- Step 4: download ---------------------------------------------------------

@contextmanager
def ollama_client(host: str):
    """Yield a connected Ollama client, starting `ollama serve` only if needed and stopping it afterwards.

    Yields None (after printing the reason) when Ollama cannot be reached or started."""
    # needs the packages that ensure_packages installed
    from imagent.vlm.client import MIN_OLLAMA_VERSION, is_local, is_reachable, make_client, server_version, start_server, stop_server
    client = make_client(host)

    started = None
    try:
        if not is_reachable(client):
            if not is_local(host):
                say(f"✗ Cannot reach Ollama at {host}.", RED)
                yield None
                return
            if not app_config.ollama_binary():
                say(f"✗ The `ollama` program was not found. Run {app_config.SETUP_COMMAND} again to install it.", RED)
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
                f"then run {app_config.SETUP_COMMAND} again.", RED)
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
            problems[name] = f"cannot be checked because your Ollama is too old. Update Ollama, then run {app_config.SETUP_COMMAND} again."
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

    Returns (device settings, None), or (None, problems) when a model does not run on this computer: ``problems`` maps the name
    of the model that failed (the vision model, or the embedding model when that is the one) to what is wrong with it."""
    from imagent.vlm.calibration import NotPossible, calibrate
    say("Measuring this computer with the vision model. This runs by itself and can take a few minutes…")
    try:
        device = calibrate(client, vlm_model, embedding_model, lambda message: say(f"  {message}"))
    except NotPossible as exc:
        return None, {exc.model or vlm_model: str(exc)}
    except Exception as exc:
        return None, {vlm_model: f"could not be measured ({type(exc).__name__}: {exc})."}
    say(f"✓ Pictures up to {device['image_long_edge']} pixels, a memory window of {device['num_ctx']}, "
        f"waiting up to {device['timeout_seconds']} seconds for a reply.", "fg:ansigreen")
    if device["one_model_at_a_time"]:
        say("⚠ The vision and embedding models do not fit in memory together on this computer. Imagent will load one at a time, "
            "which makes indexing a little slower.", "fg:ansimagenta")
    return device, None


def _remove_failed(client, rejected: set, fixed: dict) -> None:
    """Take off this computer the models that did not work, so they do not keep using its disk.

    Only what Imagent itself downloaded or copied (the list in install_record.json, which the uninstaller uses too) is removed: a
    model that was on the computer before is never touched. Nothing is removed that a model which is kept needs.
    ``rejected`` holds the names as typed, ``fixed`` maps each typed name to the name Imagent uses."""
    record = app_config.load_install_record()
    ours = {m for m in record.get("ollama_models", []) if isinstance(m, str)}
    pins = record["pinned_models"] if isinstance(record.get("pinned_models"), dict) else {}
    kept = set()
    for typed, used in fixed.items():
        if typed not in rejected:
            kept |= {_normalize(typed), used, pins.get(used)}
    removed = set()
    for typed in sorted(rejected):
        names = [name for name in (fixed.get(typed), _normalize(typed)) if name in ours and name not in kept]
        for name in names:
            try:
                client.delete(name)
                removed.add(name)
            except Exception as exc:
                say(f"⚠ Could not remove {name} ({type(exc).__name__}: {exc}). You can remove it with: ollama rm {name}", "fg:ansimagenta")
        if any(name in removed for name in names):
            say(f"✓ Removed {typed}, which did not work on this computer, to free its disk space.", "fg:ansigreen")
    if removed:
        app_config.save_install_record({**record, "ollama_models": [m for m in record.get("ollama_models", []) if m not in removed],
                                        "pinned_models": {name: base for name, base in pins.items() if name not in removed}})


def download_and_verify(host: str, vlm: str, embedding: str, text_model: str):
    """Download all models and verify them. Returns (settings, problems); settings is None on failure.

    ``problems`` maps the name of each model that did not work, as typed, to what is wrong with it."""
    with ollama_client(host) as client:
        if client is None:
            return None, {}
        try:
            _pull_models(client, [vlm, embedding, text_model])
        except Exception as exc:
            print()
            say(f"✗ Download failed: {exc}", RED)
            return None, {}
        from imagent.vlm.client import pin_model
        fixed = {}  # each model as the user typed it -> the name Imagent will use; one copy when two roles share a model
        for typed in dict.fromkeys((vlm, embedding, text_model)):
            original = _normalize(typed)
            try:
                fixed[typed], _ = pin_model(client, original)
                app_config.record_pinned_model(fixed[typed], original)
                say(f"✓ Saved a copy of {original} that stays exactly as it is now, even if the model is updated later.", "fg:ansigreen")
            except Exception as exc:
                fixed[typed] = original
                say(f"⚠ Could not save a fixed copy of {original} ({type(exc).__name__}: {exc}). Imagent will use it as it is "
                    "and stop if it ever changes.", "fg:ansimagenta")
        say("Checking that the models work…")
        try:
            settings, problems = verify_models(client, fixed[vlm], fixed[embedding], fixed[text_model])
        except Exception as exc:
            say(f"✗ Could not check the models: {exc}", RED)
            return None, {}
        typed_as = {used: typed for typed, used in fixed.items()}
        problems = {typed_as.get(name, name): reason for name, reason in problems.items()}
        if settings is not None:
            device, measure_problems = measure_computer(client, settings["vlm_model"], settings["embedding_model"])
            if measure_problems:
                settings = None
                problems = {typed_as.get(name, name): reason for name, reason in measure_problems.items()}
                for name, reason in problems.items():  # said before the removal below, so the reason comes first
                    say(f"✗ '{name}' {reason}", RED)
                _remove_failed(client, set(problems), fixed)
                return None, problems
            settings["device"] = device
    for name, reason in problems.items():
        say(f"✗ '{name}' {reason}", RED)
    return settings, problems


# --- Choosing models (first run only) ------------------------------------------

def choose_models(saved_host: str):
    """Input boxes -> verify on ollama.com -> specs -> download -> verify.

    Returns (settings, host). The caller locks the settings into the database."""
    from imagent.vlm.client import is_local  # needs the packages that ensure_packages installed
    step("Models")
    host = saved_host or DEFAULT_HOST
    vlm = embedding = text_model = ""
    failed: dict = {}  # every model that was tried here and did not work, with the reason: the AI prompt tells the AI
    while True:
        vlm, embedding, text_model = ask_models(vlm, embedding, text_model, failed)
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
        failed.update(rejected)
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
    from imagent.vlm.client import is_pinned, pinned_name
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
            "so the original cannot be restored. Uninstall Imagent and set it up again to use the models that are installed "
            "(this clears the index).", RED)
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
                    say("    ⚠ This is not the same version it was when you set up Imagent, so Imagent will not run until the original is back.", "fg:ansimagenta")
            for name in others:
                say(f"  {name:<{width}}  {_fmt_bytes(installed[name][0]):>8}")
            if not installed:
                say("  (no models installed)")
            say("")
            say("Note: the models are locked once set. Each embedding model produces different vectors for",
                "fg:ansibrightblack")
            say("the same text, so switching models would break all existing image searches. To use other",
                "fg:ansibrightblack")
            say("models, uninstall Imagent and set it up again (this clears the index).", "fg:ansibrightblack")
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
    from imagent.image_processing import sample_pictures
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
    """Setup was run before: measure this computer if it never was (the models are locked, so they never change and need no new measurement)."""
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
        if digest is None or (device is not None and device["one_model_at_a_time"] is not None):
            return  # a missing model was offered for reinstall above; a measured computer stays measured
        device, problems = measure_computer(client, settings["vlm_model"], settings["embedding_model"])
    if problems:
        name, reason = next(iter(problems.items()))
        fail(f"'{name}' {reason}")
    conn = init_db(str(app_config.DB_PATH))
    try:
        with conn:
            set_device_settings(conn, device, digest)
    finally:
        conn.close()


def _validate_keep_deleted(value: str):
    return True if value.strip().isdigit() else "Enter a whole number, 0 or more."


def ask_keep_deleted() -> int:
    """Explain what remembering deleted pictures is for, and ask how many to remember: one of ``KEEP_DELETED_CHOICES``, or a number
    typed in (0 remembers none)."""
    say("When you delete a picture, Imagent forgets it at the next scan. It can instead remember what it learned about the picture:")
    say("its description and search data (not the picture, and not where it was).")
    say("If the same picture comes back later (restored from a backup, copied, or moved into a folder you add later), Imagent")
    say("recognises it by its pixels and uses what it remembers, so it does not have to be analysed again. That saves time when")
    say("indexing. When there are more than the number you choose, the oldest are forgotten first.\n")
    choices = [questionary.Choice(f"{number} deleted pictures", value=number) for number in KEEP_DELETED_CHOICES]
    choices.append(questionary.Choice("Enter a number", value=ENTER_A_NUMBER))
    question = questionary.select(
        "How many deleted pictures should Imagent remember?", choices=choices,
        default=next(c for c in choices if c.value == KEEP_DELETED_DEFAULT), qmark=QMARK, pointer=POINTER, style=STYLE,
    )
    # questionary also marks the default as "selected", which the style underlines for good. Keep only the pointer on it.
    control = next(w.content for w in question.application.layout.find_all_windows() if hasattr(w.content, "selected_options"))
    control.selected_options.clear()
    choice = ask(question)
    if choice != ENTER_A_NUMBER:
        return choice
    return int(ask(questionary.text("How many deleted pictures should Imagent remember? (0 remembers none)",
                                    validate=_validate_keep_deleted, qmark=QMARK, style=STYLE)).strip())


def choose_keep_deleted() -> None:
    """Ask what happens to the data of a deleted picture, and store the answer. When it was answered before, it is shown and only asked again if the user wants."""
    conn = init_db(str(app_config.DB_PATH))
    try:
        current = get_keep_deleted(conn)
        step("Deleted pictures")
        if current is not None:
            say(f"Imagent remembers the last {current} deleted pictures." if current else "Imagent does not remember deleted pictures.")
            if not ask(questionary.confirm("Change it?", default=False, qmark=QMARK, style=STYLE)):
                return
        keep = ask_keep_deleted()
        with conn:
            set_keep_deleted(conn, keep)
    finally:
        conn.close()
    say(f"✓ Imagent will remember the last {keep} deleted pictures." if keep else "✓ Imagent will not remember deleted pictures.",
        "fg:green")


# --- Main ---------------------------------------------------------------------

def main() -> None:
    make_output_safe()
    if not sys.stdin.isatty():
        fail(f"Setup needs an interactive terminal. Run it directly: {app_config.SETUP_COMMAND}")

    step("Your computer")
    say(system_details())

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
    from imagent.vlm.client import is_local
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
        choose_keep_deleted()
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
    choose_keep_deleted()
    say(f"\n✓ Setup complete. Your models are locked in.", "fg:green")


