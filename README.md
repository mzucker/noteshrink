noteshrink
==========

Convert scans of handwritten notes to beautiful, compact PDFs -- see full writeup at <https://mzucker.github.io/2016/09/20/noteshrink.html>

## Requirements

 - Python 2 or 3

Python libraries: (use requirements.txt)
 - NumPy 1.10 or later
 - SciPy
 - Image module from PIL or Pillow

Also:
 - ImageMagick (for "convert")
   - on macOS: `brew install imagemagick` (requires Homebrew)
   - on Debian/Ubuntu: `sudo apt-get install imagemagick`

## Usage

```
./noteshrink.py IMAGE1 [IMAGE2 ...]
```

Building the examples (already in `example_output`):

```
make
```

## Packages
Packages are available for:
 - [Arch Linux (AUR)](https://aur.archlinux.org/packages/noteshrink/)

## Derived works

*Note:* Projects listed here aren't necessarily tested or endorsed by me -- use with care!

  - [Web-based (Django) front-end](https://github.com/delneg/noteshrinker-django)
