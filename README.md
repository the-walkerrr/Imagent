# Imagent

Search your local images in plain English. Imagent analyses each image with a
local vision-language model (via [Ollama](https://ollama.com)), reads any text in
it with OCR, stores everything in a SQLite database, and answers natural-language
queries such as "the receipt from the coffee shop" with the best matching images.
Everything runs on your machine.

## Requirements

- Python 3.9 or newer, with SQLite 3.24 or newer (every current Python has it). This is the only thing
  to install yourself; the launcher checks it and tells you how to get it if it is missing or too old.
- [Ollama](https://ollama.com) 0.5 or newer (installed locally, or reachable over the network).
  `install.py` installs or updates it for you, after asking.
- A vision-language model and an embedding model available in Ollama
  (`install.py` helps you choose and download them)

## Getting started

```bash
sh install.sh          # macOS / Linux
install.bat            # Windows (double-click it, or run it in a terminal)
```

The launcher is the only thing you need to run to set Imagent up. It finds a Python 3.9 or newer and
starts `install.py` with it; if there is none, it prints how to install one for your OS and stops
(open a new terminal after installing). It changes nothing else. If you already have Python you can run
`python3 install/install.py` (or `py install/install.py` on Windows) directly.

`install.py` is an interactive first-run script that works
the same on macOS, Windows and Linux. It checks your Python version, creates a `.venv`, and
checks the packages in `requirements.txt` against it. Anything missing or too old is listed and installed
only after you agree, and only into `.venv`, so your own Python and its packages are never changed.
It then downloads the small text-reading (OCR) models, checks Ollama (offering to install or
update it if it is missing or older than 0.5), lets you pick three models, downloads and verifies them,
and locks them in the database:

| Model | Used for | When |
| --- | --- | --- |
| VLM (vision) | Describing each image | Indexing |
| Embedding | Turning descriptions and queries into vectors | Indexing and searching |
| Judge (text) | Understanding the query (fixing typos and drafting a description of the image you want); it never sees an image and may be the same model as the VLM | Searching |

Indexing and searching never need all three at once, so each only has to fit in memory with
the embedding model. Indexing unloads the VLM when it finishes. **The models cannot be changed afterwards**
(each embedding model produces different vectors, so a change would break every
existing search). Running `install.py` again only shows their status and reinstalls a
model that was removed from Ollama.

Then:

```bash
sh run.sh          # macOS / Linux
run.bat            # Windows (double-click it, or run it in a terminal)
```

It asks what to do:

1. Search your images: asks for a query and shows the closest images.
2. Scan folders and index images: asks for a folder, finds new/moved/changed/deleted images, then offers to index everything waiting.
3. Uninstall Imagent (shown in red): starts the uninstaller described below.

Search and scan run inside `.venv`; the uninstaller does not, because it deletes `.venv`.

You can also run `search/search.py` or `scan_and_index/scan_and_index.py [folder]` yourself with the Python from `.venv`.

You can scan several folders; they share one index and one search. Scanning a folder only
updates that folder's images. Folders that contain or lie inside an already scanned folder are
rejected.

A scan protects the index from a folder it cannot see properly:

- A subfolder that cannot be read (permissions, a share that dropped) keeps its indexed images.
- If all of a folder's images, or most of them (at least 10, and more than half), seem to be gone
  (a drive that is not mounted, for example), `scan_and_index.py` asks before removing them from the index.
- Not scanned: hidden folders (`.git`, `.Trash`, ...), `node_modules`, photo libraries
  (`*.photoslibrary`), Synology `@eaDir`, the recycle bin, macOS `._*` companion files, empty files,
  and files whose name is not valid text (they are listed so you can rename them).

Indexing is safe to interrupt, including by a crash or a power cut. Each image is saved in one
database transaction, so it is either fully indexed or not at all, and the next run carries on
with what is left. Only one indexing run can be active at a time. If the disk is full, the
database is locked or Ollama keeps failing, indexing stops with a message, the files stay queued
and none of their attempts are used up.

Imagent stores two files, in a per-user folder that must be on a local disk:

| OS | Folder |
| --- | --- |
| macOS | `~/Library/Application Support/Imagent` |
| Windows | `%LOCALAPPDATA%\Imagent` (`C:\Users\<you>\AppData\Local\Imagent`) |
| Linux | `~/.local/share/imagent` |

- `imagent.db`: the index, the scanned folders and the locked models.
- `config.json`: your preferences (the Ollama address). It is replaced atomically, so it cannot be left half-written.
- `install_record.json`: the Ollama models `install.py` downloaded, so that uninstalling removes those and nothing else.

The database has a schema version and upgrades itself when you update Imagent (in one transaction,
so a failed upgrade changes nothing). A database made by a newer Imagent is refused rather than
risked. If something unexpected goes wrong, the details are saved to `last_error.log` in the same
folder; attach that file when you report it.

## Uninstalling

```bash
sh uninstall/uninstall.sh        # macOS / Linux
uninstall\uninstall.bat          # Windows
```

This takes Imagent off the computer and leaves only the project's source files. It first shows what you
would lose: how many images are indexed, the scanned folders, how long the analysis took, and how much
disk space each part uses. Then it asks three different questions (type `yes`, type the number of
indexed images, type `uninstall imagent`), and a wrong answer to any of them cancels everything.

It removes:

- the Ollama models `install.py` downloaded. A model you already had in Ollama is never removed. Models
  downloaded before Imagent kept a record (`install_record.json`) are listed so you can remove them yourself
  with `ollama rm <name>`;
- Imagent's data folder (the table above): the index, `config.json`, the logs;
- the `.venv` folder (the packages and the text-reading models) and every `__pycache__` folder.

It never touches your image files, Ollama itself, or any other Ollama model. When it finishes, delete the
project folder yourself if you no longer want the source. Like `install.sh`, the launcher needs a Python
3.9 or newer, and it will not run from inside `.venv` (run `deactivate` first).

## Project layout

| Path | Purpose |
| --- | --- |
| `install.sh`, `install.bat` | Setup launchers; they start `install/install.py` |
| `run.sh`, `run.bat` | Menu launchers: choose search, scan and index, or uninstall, then they run it |
| `install/`, `search/`, `scan_and_index/`, `uninstall/` | One folder per command-line flow: its Python script (`uninstall/` also holds its `.sh` / `.bat` launcher) |
| `tests/` | Automated tests (standard library only; need no Ollama) |
| `config/` | Every setting (no environment variables are read), the per-OS storage paths and prompt loading |
| `prompts/` | Prompt templates sent to the model |
| `db/` | SQLite connection, database schema (`schema.sql`), schema versioning and queries |
| `image_processing/` | Hashing, metadata/EXIF, OCR, preprocessing, colour features |
| `text_processing/` | Cleaning and search-text formatting |
| `pipeline/` | Folder discovery, validation, serialisation, error logging, progress display, run lock, command-line wrapper |
| `vlm/` | Ollama client and readiness check, image analysis, embeddings, query understanding, Pydantic models for the model replies (`response_format.py`) |

## Tests

```bash
python -m unittest        # run from the project folder, with the Python from .venv
```

The tests use a temporary folder for the database and never touch your real index or call Ollama.

## Configuration

Every setting is a constant in `config/__init__.py`; no environment variables or `.env` file are read.
Edit the file to change one. The ones you are most likely to tune:

| Setting | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_TIMEOUT` | `300` | Seconds to wait for Ollama before a request fails |
| `OLLAMA_USE_MMAP` | `False` | Load each model fully into memory instead of paging it from disk |
| `IMAGE_MAX_LONG_EDGE` | `1024` | Longest side (px) of the copy sent to the model |
| `IMAGE_JPEG_QUALITY` | `85` | JPEG quality of the copy sent to the model |
| `BACKLOG_MAX_ATTEMPTS` | `3` | Times a file may fail to index before it is left alone until it changes |
| `UNCERTAIN_TOP_K` | `3` | The most results returned per search |
