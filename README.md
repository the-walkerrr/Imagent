# Imagent (Beta)

**Local image indexing and search tool.**

- **Private:** everything runs on your computer, and your images never leave it.
- **Offline:** after setup, no internet connection is needed.
- **Cross-platform:** macOS, Linux and Windows.
- **Wide format support:** JPG, PNG, WebP, HEIC/HEIF, BMP, TIFF, GIF and AVIF.

## Requirements

Python 3.9 or newer, [pipx](https://pipx.pypa.io) (or `uv`) and [Git](https://git-scm.com). Setup offers to install [Ollama](https://ollama.com) if you do not have it.

## Install

Install the `imagent` command (once):

```bash
pipx install git+https://github.com/the-walkerrr/Imagent.git
```

Then set it up (once). It checks Ollama, downloads the three models, locks them and measures your computer:

```bash
imagent setup
```

## Use

```bash
imagent search "red car at night"   # show the pictures that best match a description
imagent search                      # keep searching until you type exit
imagent scan ~/Pictures             # find new or changed pictures in a folder, then index them
imagent scan                        # the same, choosing the folder when asked
imagent failed                      # list the pictures that could not be indexed, and why
imagent                             # open a menu (arrow keys and Enter) with the same choices
```

`imagent --help` lists the commands. To index: run `imagent scan` with a folder, then `imagent search`.

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
imagent uninstall      # deletes the index, the data folder and the models Imagent downloaded (three confirmations)
pipx uninstall imagent # removes the program itself
```

Your pictures, Ollama itself and any model you had before Imagent are never touched.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `Imagent is not set up yet` | Run `imagent setup`. |
| `No indexed images found` | Run `imagent scan` with a folder first. |
| `The ... model ... is not the same version` | The models are locked. Put the original version back (`imagent setup` restores it when it can), or uninstall and set up again, which clears the index. |
| `...is on a network drive` / `...external or removable drive` | Copy the pictures to your computer's own disk. |

## Development

```bash
git clone https://github.com/the-walkerrr/Imagent.git && cd Imagent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip   # editable installs need pip 21.3 or newer
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest                    # run the tests
git config core.hooksPath .githooks             # run the tests before every commit
```

```
src/imagent/
  cli.py  __main__.py   the `imagent` command
  commands/             one module per command: search, scan, failed, setup, uninstall, menu
  config.py             every setting
  db/  vlm/  pipeline/  image_processing/  text_processing/  os_specific/  prompts/
tests/
```
