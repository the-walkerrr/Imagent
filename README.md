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
2. `run` opens a menu: **1** Search, **2** Scan and index, **3** Try again the images that could not be indexed,
   **4** Re-embed images, **5** Uninstall.
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
- lets you pick three models (vision, embedding, text) and downloads them;
- measures your computer with the vision model, by itself and with no questions: it tries pictures from large to small
  until one works, and stores the largest picture size, the memory window, whether the model is paged in from disk and
  how long to wait for a reply, all as measured here, not guessed. Running install again measures again if the vision
  model changed;
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
done: an image is either indexed completely or set aside with the reason.

| Trouble | What Imagent does |
| --- | --- |
| The model's reply is cut off | Asks again with different sampling, then with a higher output limit, step by step |
| The reply does not fit the schema | Tells the model what is wrong and asks again |
| No reply in time, or Ollama stopped | Waits longer; or restarts Ollama or frees memory, and asks again once |
| The text in the image cannot be read | Reads it in tiles, then again with the engine reloaded |
| The image is damaged, unsupported or too large for the free memory | Sets it aside at once, with the reason |
| The file is still being copied | Leaves it for the next run |
| The embedding text is too long | Splits it and averages the parts; nothing is cut off |

An image that is set aside is queued again by itself when the file changes, or (for the model's replies) when the
model or the prompt changes. To try it again yourself, choose **Try again the images that could not be indexed**.
Indexing stops, and says why, when the same trouble keeps happening: Ollama or the machine failing several images
in a row, files that cannot be read one after another (a drive that went away), or most replies needing a retry
(which points to the model or the prompt).

How large an image may be is not a fixed number: it depends on the memory free at that moment, and a very large
image is read more gently (a JPEG is shrunk while it is decoded, its text is read in tiles). To set a hard limit,
add `"max_megapixels": 100` to `config.json`.

### If the embedding model changed

Imagent refuses to index or search if the installed embedding model is not the version the stored vectors were made
with, because the vectors could not be compared. Choose **Re-embed images**: it shows how long it will take, makes all
the vectors again from what is stored (the vision model is not asked again) and saves the model as a copy of its own.

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
run.sh, run.bat              Menu: search, scan and index, try again, re-embed, uninstall
install/  search/  scan_and_index/  retry_failed/  reembed/  uninstall/    One folder per command
config/                      Settings
db/  vlm/  image_processing/  text_processing/  pipeline/  prompts/    Shared code
tests/                       Automated tests
```

## Tests

```bash
python -m unittest      # run from the project folder, with the Python from .venv
```

The tests use a temporary folder and never touch your real index or call Ollama. The real photos they use are in
`tests/fixtures/images.zip`, with where each came from and its licence (all public domain or CC0) in its `manifest.json`.
