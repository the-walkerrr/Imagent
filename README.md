# Imagent (Beta)

**Local image indexing and search tool.**

- **Private:** everything runs on your computer, and your images never leave it.
- **Offline:** after setup, no internet connection is needed.
- **Cross-platform:** macOS, Linux and Windows.
- **Wide format support:** JPG, PNG, WebP, HEIC/HEIF, BMP, TIFF, GIF and AVIF.

## Requirements

Python 3.9 to 3.14 and [Git](https://git-scm.com). Setup offers to install [Ollama](https://ollama.com) if you do not have it.

## Install

```bash
git clone https://github.com/the-walkerrr/Imagent.git && cd Imagent
python3 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip        # editable installs need pip 21.3 or newer
python -m pip install -e .
```

This adds the `imagent` command inside the virtual environment, and your edits to the code take effect straight away. Activate the environment (`source .venv/bin/activate`) in each new terminal before using `imagent`. If your default Python is newer than 3.14, create the environment with Python 3.12 instead: `python3.12 -m venv .venv`.

Then set it up (once). It checks Ollama, downloads the three models, locks them, measures your computer and asks how many deleted pictures to remember (see below):

```bash
imagent setup
```

## Use

```bash
imagent search "red car at night"   # show the pictures that best match a description
imagent search                      # keep searching until you type exit
imagent search --full "red car"     # show each result's full description instead of its one-line summary
imagent scan add ~/Pictures         # add a folder, find its new or changed pictures, then index them
imagent scan                        # scan every folder you added, then index what is pending (asks for a folder if none is added yet)
imagent failed                      # list the pictures that could not be indexed, and why
imagent forget                      # delete what Imagent remembers about deleted pictures (two confirmations)
imagent                             # open a menu (arrow keys and Enter) with the same choices
```

`imagent --help` lists the commands. To index: run `imagent scan add` with a folder (run it again for each folder you want), then `imagent search`. Run `imagent scan` later to bring all of them up to date.

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

| OS | Folder |
| --- | --- |
| macOS | `~/Library/Application Support/Imagent` |
| Windows | `%LOCALAPPDATA%\Imagent` |
| Linux | `~/.local/share/imagent` |

Holds `imagent.db`, `config.json`, the logs and `last_error.log`.

## Uninstall

```bash
imagent uninstall      # deletes the index, the data folder, the models Imagent downloaded and Ollama if Imagent installed it (three confirmations)
pipx uninstall imagent # removes the program itself
```

Your pictures, Ollama itself and any model you had before Imagent are never touched.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `Imagent is not set up yet` | Run `imagent setup`. |
| `No indexed images found` | Run `imagent scan add` with a folder first. |
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
