# Imagent (Beta)

**Local image indexing and search tool.**

- **Private:** everything runs on your computer, and your images never leave it.
- **Offline:** after setup, no internet connection is needed.
- **Cross-platform:** macOS, Linux and Windows.
- **Wide format support:** JPG, PNG, WebP, HEIC/HEIF, BMP, TIFF, GIF and AVIF.

## Requirements

Python 3.9 to 3.14 and [Git](https://git-scm.com). You do not need to install [Ollama](https://ollama.com) yourself: setup installs its own copy for Imagent when the computer has none (see below).

## Install

```bash
git clone https://github.com/the-walkerrr/Imagent.git && cd Imagent
python3 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip        # editable installs need pip 21.3 or newer
python -m pip install -e .
```

This adds the `imagent` command inside the virtual environment, and your edits to the code take effect straight away. Activate the environment (`source .venv/bin/activate`) in each new terminal before using `imagent`. If your default Python is newer than 3.14, create the environment with Python 3.12 instead: `python3.12 -m venv .venv`.

Then set it up (once). It checks Ollama (and installs it if needed), downloads the three models, locks them, measures your computer and asks how many deleted pictures to remember (see below):

```bash
imagent setup
```

## Use

```bash
imagent search "red car at night"   # show the pictures that best match a description
imagent search                      # keep searching until you type exit
imagent search --full "red car"     # show each result's full description instead of its one-line summary
imagent add ~/Pictures         # add a folder, then choose what to do next: scan and index it, add another, delete folders, search
imagent add                    # the same, picking the folder from suggestions as you type (macOS; elsewhere you type the path)
imagent scan                        # scan every folder you added, then index what is pending (asks for a folder if none is added yet)
imagent delete ~/Pictures           # remove a folder from the scan list and what was indexed from it (several folders can be given; quote a path with spaces)
imagent delete                      # the same, ticking the folders to remove from the scan list
imagent failed                      # list the pictures that could not be indexed, and why
imagent forget                      # delete what Imagent remembers about deleted pictures (two confirmations)
imagent help                        # explain each command briefly
imagent                             # open a menu (arrow keys and Enter) with the same choices
```

`imagent --help` lists the commands. To index: run `imagent add` with a folder, or without one to pick it from a list, then choose "Scan and index this folder" (the menu comes back after each choice, so you can add more folders), then `imagent search`. Run `imagent scan` later to bring all of them up to date.

## Pictures that come back

A picture is recognised by its pixels, whatever its name, folder or embedded details (EXIF) are. The vision model looks at each picture once:

- **Copies and renames:** a new file with the same pixels as a picture already indexed gets that picture's description and search data again, so it is not analysed a second time. A renamed or moved file keeps its index.
- **Deleted pictures:** when a scan finds that a picture was deleted, Imagent can remember what it learned about it (its description and search data; not the picture, and not where it was). If the same picture comes back later (restored from a backup, copied, or moved into a folder you add later), it is not analysed again. Search never shows remembered pictures.

Setup asks how many deleted pictures to remember: 100, 500, 1000, 2000, or a number you type (0 remembers none). When there are more than that, the oldest are forgotten first. Run `imagent setup` again to see your choice and change it. Until you have answered, nothing is remembered. To delete what is remembered now, run `imagent forget` (or choose it in the menu); your indexed pictures and your choice are not touched.

A description is only used again when it was made with the same models, prompt and text-reading engine as a new one would be; otherwise the picture is analysed again.

## When a model does not fit your computer

Setup tries each model for real. If one cannot run here, or cannot write a description in the form Imagent needs, setup says which one, removes what it downloaded for it (a model that was already on your computer is never removed) and asks again; the prompt you copy for an AI chat then lists the models that already failed.

If a picture cannot be indexed because Ollama ran out of memory, stopped or did not reply in time, `imagent scan` lists those pictures and why, and asks whether to try them again with lower settings (a longer wait, the model paged in from disk, a smaller picture). The lower settings apply only to those pictures, each by what went wrong with it; your usual settings do not change. The default answer is no.

## The models are locked

`imagent setup` saves a private copy of each of the three models (vision, text and embedding), so a later `ollama pull` cannot change them. Different embedding models give different vectors for the same text, so changing a model would break every search of what is already indexed. To use other models, uninstall Imagent and set it up again (this clears the index).

## Where your data lives

Everything is in `~/.imagent` (on Windows, `.imagent` in your user folder), on every OS.

Holds `imagent.db`, `config.json`, the logs and `last_error.log`.

## Ollama

If `imagent setup` finds no Ollama, or one older than Imagent needs, it downloads Ollama's official build for your computer into a folder of its own next to the data folder, with no administrator rights, no package manager and no change to `PATH`. The download is about 190 MB on macOS, 1.5 to 1.6 GB on Linux and 0.4 to 1.6 GB on Windows, and it unpacks to about 0.6 GB on macOS and up to about 2.3 GB on Linux and Windows, so keep up to about 4 GB free while it installs. Every file is checked against the sums in Ollama's own list for that release, and that list is checked against a checksum built into Imagent. This copy is used only by Imagent, and it keeps its models there too. An Ollama you already have, if it is recent enough, is used as it is and nothing is installed.

Imagent's own Ollama is in `~/.imagent-tools/ollama` (on Windows, `.imagent-tools\ollama` in your user folder), on every OS.

The copy has no GPU drivers, so an NVIDIA GPU is used only if its driver is already installed (the build itself holds the CUDA libraries). Setup looks at your GPU and adds the extra files Ollama publishes for it: for an AMD GPU (Linux or Windows) the ROCm files, about 1 GB more to download on Linux and 0.3 GB on Windows, and for an NVIDIA Jetson board the files for its JetPack version, about 0.3 GB. They are unpacked over the same folder, and they take more disk space than they download. A computer with an NVIDIA GPU gets no extra files, even when it has an AMD one too, and neither does a Mac.

## Uninstall

```bash
imagent uninstall      # deletes the index, the data folder and the models Imagent downloaded (three confirmations), then asks whether to remove Imagent's own Ollama too
pipx uninstall imagent # removes the program itself
```

Your pictures, an Ollama you installed yourself and any model you had before Imagent are never touched. Imagent's own Ollama is removed only if you answer yes (the default is to keep it); you can delete its folder yourself at any time.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `Imagent is not set up yet` | Run `imagent setup`. |
| `No indexed images found` | Run `imagent add` with a folder first. |
| `The ... model ... is not the same version` | The models are locked. Put the original version back (`imagent setup` restores it when it can), or uninstall and set up again, which clears the index. |
| `...is on a network drive` / `...external or removable drive` | Copy the pictures to your computer's own disk. |

## Development

```bash
python -m unittest                    # run the tests (with the environment from Install active)
git config core.hooksPath .githooks   # run the tests before every commit
```

```
src/imagent/
  cli.py  __main__.py   the `imagent` command
  commands/             one module per command: search, scan, failed, setup, uninstall, menu
  config.py             every setting
  db/  vlm/  pipeline/  image_processing/  text_processing/  os_specific/  prompts/
tests/
```
