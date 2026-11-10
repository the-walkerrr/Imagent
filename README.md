# Imagent (Beta)

**Local image indexing and search tool.**

- **Private:** everything runs on your computer, and your images never leave it.
- **Offline:** after setup, no internet connection is needed.
- **Cross-platform:** macOS, Linux and Windows.
- **Wide format support:** JPG, PNG, WebP, HEIC/HEIF, BMP, TIFF, GIF and AVIF.

## Requirements

[Git](https://git-scm.com) and Python 3.9 or newer. The installer sets up everything else.

## Install and run

**macOS / Linux**

Clone and set up (once):

```bash
git clone https://github.com/the-walkerrr/Imagent.git && cd Imagent && sh install.sh
```

Open the menu:

```bash
sh run.sh
```

**Windows** (Command Prompt)

Clone and set up (once):

```bat
git clone https://github.com/the-walkerrr/Imagent.git && cd Imagent && install.bat
```

Open the menu:

```bat
run.bat
```

## Menu

`run` opens a menu (arrow keys and Enter):

| Choice | Script |
| --- | --- |
| Search your images | `search/search.py` |
| Scan folders and index images | `scan_and_index/scan_and_index.py` |
| Show the images that could not be indexed (only when there are some) | `show_failed/show_failed.py` |
| Re-embed images (only when the embedding model changed) | `reembed/reembed.py` |
| Uninstall Imagent | `uninstall/uninstall.sh` / `uninstall\uninstall.bat` |

To index: choose **Scan folders and index images**, give it a folder, then choose **Search your images**.

## Where your data lives

| OS | Folder |
| --- | --- |
| macOS | `~/Library/Application Support/Imagent` |
| Windows | `%LOCALAPPDATA%\Imagent` |
| Linux | `~/.local/share/imagent` |

Holds `imagent.db`, `config.json`, the logs and `last_error.log`.

## Troubleshooting

| Message | What to do |
| --- | --- |
| `Imagent is not set up yet` | Run `sh install.sh` (or `install.bat`). |
| `No indexed images found` | Scan and index a folder first. |
| `The embedding model ... is not the same version` | Choose **Re-embed images**. |
| `...is on a network drive` / `...external or removable drive` | Copy the pictures to your computer's own disk. |
