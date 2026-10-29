# Imagent (Beta)

Search your local images in plain English. Imagent describes each image with a local vision model
(through [Ollama](https://ollama.com)), reads any text in it with OCR, and stores everything in a local
database. Then a query like "the receipt from the coffee shop" finds the best matching images.

- Runs entirely on your computer; your images never leave it.
- Works on macOS, Linux and Windows.
- Handles JPG, PNG, WebP, HEIC/HEIF, BMP, TIFF, GIF and AVIF.

## Requirements

- [Git](https://git-scm.com) and Python 3.9 or newer. Everything else (packages, Ollama, models) is set up by the installer.
- Disk space and memory for the models you pick. A larger model is more accurate but slower.

## Quick start

Copy and paste one command into a terminal.

**macOS / Linux**

```bash
git clone https://github.com/the-walkerrr/Imagent.git && cd Imagent && sh install.sh && sh run.sh
```

**Windows** (Command Prompt)

```bat
git clone https://github.com/the-walkerrr/Imagent.git && cd Imagent && call install.bat && call run.bat
```

What happens:

1. `install` asks a few questions and sets Imagent up (once). Answer them and wait for it to finish.
2. `run` opens a menu: **1** Search, **2** Scan and index, **3** Uninstall.
3. Choose **2**, give it a folder of images, and let it index them.
4. Run `run` again, choose **1**, and search.

## Usage

After the first setup, start Imagent from the project folder with:

```bash
# macOS / Linux
sh run.sh 
```
```bash
# Windows
run.bat          
```

If something is not set up, `run` tells you to run `sh install.sh` (or `install.bat`) first.

## How it works

### Install

`install.sh` / `install.bat` is an interactive setup. It:

- creates a private `.venv` and installs the required packages into it (your own Python is not changed);
- checks Ollama and offers to install or update it;
- lets you pick three models (vision, embedding, text) and downloads them.

The models cannot be changed afterwards, because every embedding model produces different vectors.
Running install again only shows their status.

### Scan and index

Choose **Scan folders and index images**. Give it a folder, and it finds new, moved, changed and deleted
images, then offers to index them. You can scan several folders; they share one index. Indexing is safe to
interrupt: run it again and it carries on with what is left.

### Search

Choose **Search your images** and type what you are looking for. It shows the three closest images.
Type `exit` to quit. Scan and index at least one folder first.

### Uninstall

Choose **Uninstall Imagent** (or run `sh uninstall/uninstall.sh` / `uninstall\uninstall.bat`). It shows what
will be lost and asks three times before deleting anything. It removes the Ollama models Imagent downloaded,
Imagent's data (index, settings, logs) and `.venv`. It never touches your images or Ollama itself.

## Where your data lives

| OS | Folder |
| --- | --- |
| macOS | `~/Library/Application Support/Imagent` |
| Windows | `%LOCALAPPDATA%\Imagent` |
| Linux | `~/.local/share/imagent` |

It holds the index (`imagent.db`), your settings (`config.json`) and the logs. If something goes wrong, the
details are in `last_error.log` there.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `Imagent is not set up yet` | Run `sh install.sh` (or `install.bat`). |
| `Cannot reach Ollama` | Start the Ollama app (or run `ollama serve`) and try again. |
| `No indexed images found` | Run `run`, choose **Scan and index**, then search again. |
| Python not found or too old | Install Python 3.9+ (`brew install python` on macOS, `winget install Python.Python.3.12` on Windows), then open a new terminal. |
| OCR models fail to download | The first run needs an internet connection once. |

## Project structure

```
install.sh, install.bat      Set up Imagent
run.sh, run.bat              Menu: search, scan and index, uninstall
install/  search/  scan_and_index/  uninstall/    One folder per command
config/                      Settings
db/  vlm/  image_processing/  text_processing/  pipeline/  prompts/    Shared code
tests/                       Automated tests
```

## Tests

```bash
python -m unittest      # run from the project folder, with the Python from .venv
```

The tests use a temporary folder and never touch your real index or call Ollama.
