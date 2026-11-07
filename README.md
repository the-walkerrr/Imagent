# Imagent (Beta)

Search your local images in plain English. Imagent describes each image with a local vision model
(through [Ollama](https://ollama.com)), reads any text in it with OCR, and stores everything in a local
database. Then a query like "the receipt from the coffee shop" finds the best matching images.

- Runs entirely on your computer; your images never leave it. After setup it needs no internet connection.
- Works on pictures stored on your computer's own disk. Folders on network drives and external drives are refused.
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
2. `run` opens a menu you move through with the arrow keys and Enter: Search, Scan and index, Uninstall.
   **Re-embed images** appears only when the embedding model was changed or deleted in Ollama, and **Show the images
   that could not be indexed** only when there are some.
3. Choose **Scan folders and index images**, give it a folder of images, and let it index them.
4. Run `run` again, choose **Search your images**, and search.

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

If Imagent is not installed yet, `run` asks whether to run the installer now (`sh install.sh` or `install.bat`).

## How it works

### Install

`install.sh` / `install.bat` is an interactive setup. It:

- creates a private `.venv` and installs the required packages into it (your own Python is not changed);
- checks Ollama and offers to install or update it;
- lets you pick three models (vision, embedding, text) and downloads them;
- measures your computer with the vision model, by itself and with no questions: it tries pictures from large to small
  until one works, and stores the largest picture size, the memory window, whether the model is paged in from disk and
  how long to wait for a reply, all as measured here, not guessed. It also checks that the embedding model runs while
  the vision model is loaded, as it does for every image; if the two do not fit in memory together, indexing loads one
  at a time (a little slower). Running install again measures again if the vision model changed;
- unpacks a set of test pictures (real photos of many kinds and sizes, and files made on the spot: huge, damaged and odd
  ones, up to about 120 MB) and checks that this computer reads them, again by itself.

The models cannot be changed afterwards, because every embedding model produces different vectors.
Setup keeps a copy of the embedding model under its own name (`imagent-pin/…`), so updating the original model
later cannot change what Imagent uses. Running install again only shows their status.

### Scan and index

Choose **Scan folders and index images**. Give it a folder, and it finds new, moved, changed and deleted
images, then offers to index them. You can scan several folders; they share one index. Indexing is safe to
interrupt: run it again and it carries on with what is left.

### When an image cannot be indexed

Every kind of trouble has its own steps, and none of them lowers the quality of the result. Nothing is stored half
done: an image is either indexed completely or listed with the reason and what you can do about it.

| Trouble | What Imagent does |
| --- | --- |
| The model's reply is cut off | Asks again with different sampling, then with a higher output limit, step by step |
| The reply does not fit the schema | Tells the model what is wrong and asks again |
| No reply in time, or Ollama stopped | Waits longer; or restarts Ollama or frees memory, and asks again once. If that fails, the image is tried again, 3 tries in all, and again on every later scan |
| The embedding model fails (usually the memory ran out beside the vision model) | Unloads every model, waits, and asks again once |
| The text in the image cannot be read | Reads it in tiles, then again with the engine reloaded |
| The image is in a format Imagent cannot open | Converts a copy to PNG with the tool the system has (`sips` on macOS, the built-in imaging components on Windows), analyses the copy and deletes it. Linux has no such tool |
| The image is damaged | Sets it aside at once, with the reason and what to do |
| The image is too large for the free memory | Leaves it for the next scan, with the reason |
| The file is still being copied | Leaves it for the next run |
| The embedding text is too long | Splits it and averages the parts; nothing is cut off |

You do not have to retry anything yourself. An image that could not be indexed is queued again by itself when the
file changes, when the model or the prompt changes (for the model's replies), or on the next scan (when the trouble was
Ollama or the memory). Only when every try is over are you told, at the end of the scan and under **Show the images
that could not be indexed**: each image with the reason and what you can do about it. An image leaves that list as soon
as it is indexed. Indexing stops, and says why, when the same trouble keeps happening: Ollama or the machine failing several images
in a row, files that cannot be read one after another (a drive that went away), or most replies needing a retry
(which points to the model or the prompt).

How large an image may be is not a fixed number: it depends on the memory free at that moment, and a very large
image is read more gently (a JPEG is shrunk while it is decoded, its text is read in tiles). To set a hard limit,
add `"max_megapixels": 100` to `config.json`.

### If the embedding model changed

Imagent refuses to index or search if the installed embedding model is not the version the stored vectors were made
with, because the vectors could not be compared. The menu then offers **Re-embed images** (it is not shown otherwise): choose it; it shows how long it will take, makes all
the vectors again from what is stored (the vision model is not asked again) and saves the model as a copy of its own.

### Search

Choose **Search your images** and type what you are looking for. It shows the three closest images.
Type `exit` to quit. Scan and index at least one folder first.

### Uninstall

Choose **Uninstall Imagent** (or run `sh uninstall/uninstall.sh` / `uninstall\uninstall.bat`). It shows what
will be lost and asks three times before deleting anything (the last time you type a random 10-character code it shows). It removes the Ollama models Imagent downloaded,
Imagent's data (index, settings, logs) and `.venv`. It never touches your images or Ollama itself.

## Where your data lives

| OS | Folder |
| --- | --- |
| macOS | `~/Library/Application Support/Imagent` |
| Windows | `%LOCALAPPDATA%\Imagent` |
| Linux | `~/.local/share/imagent` |

It holds the index (`imagent.db`), your settings (`config.json`), the logs and the test pictures (`test_pictures`). If
something goes wrong, the details are in `last_error.log` there.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `Imagent is not set up yet` | Run `sh install.sh` (or `install.bat`). |
| `Cannot reach Ollama` | Start the Ollama app (or run `ollama serve`) and try again. |
| `No indexed images found` | Run `run`, choose **Scan and index**, then search again. |
| Python not found or too old | Install Python 3.9+ (`brew install python` on macOS, `winget install Python.Python.3.12` on Windows), then open a new terminal. |
| `The text-reading (OCR) engine could not be loaded` | Run `sh install.sh` (or `install.bat`); if it keeps failing, delete the `.venv` folder first. |
| `...is on a network drive` / `...external or removable drive` | Copy the pictures to your computer's own disk and scan that folder. |
| `The embedding model ... is not the same version` | Run `run` and choose **Re-embed images**. |
| `The model's reply had to be asked for again for ... images` | The vision model or the prompt does not suit your images. Try another model or revise `prompts/analysis_prompt.txt` (and `analysis_schema.json`). |

## Project structure

```
install.sh, install.bat      Set up Imagent
run.sh, run.bat              Open the menu (menu/): search, scan and index, images that failed, re-embed (when needed), uninstall
install/  menu/  search/  scan_and_index/  show_failed/  reembed/  uninstall/    One folder per command
config/                      Settings
db/  vlm/  image_processing/  text_processing/  pipeline/  prompts/    Shared code
tests/                       Automated tests
```

## Tests

```bash
python -m unittest      # run from the project folder, with the Python from .venv
```

A pre-commit hook in `.githooks/` runs them before every commit and stops the commit if one fails. Turn it on once
per clone with `git config core.hooksPath .githooks` (it needs `.venv`, so install first).

The tests use a temporary folder and never touch your real index or call Ollama. The real photos they use are in
`tests/fixtures/images.zip`, with where each came from and its licence (all public domain or CC0) in its `manifest.json`.
