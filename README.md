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

Now it'll also work with folders passed as arguments, grabbing all images with specified extensions(by default png and jpg files) e.g.

```
./noteshrink.py -E '.png, .jpg' examples
```

Building the examples (already in `example_output`):

```
make
```

PDF command failed for windows users:

ImageMagick's Convert command is probably colliding with windows' one(https://technet.microsoft.com/en-us/library/bb490885.aspx), you can easily fix that assigning it with '-c' argument
```
python noteshrink.py -c "magick convert %i %o" IMAGE1 [IMAGE2 ...]
```

## Packages
Packages are available for:
 - [Arch Linux (AUR)](https://aur.archlinux.org/packages/noteshrink/)

## Derived works

*Note:* Projects listed here aren't necessarily tested or endorsed by me -- use with care!

  - [Web-based (Django) front-end](https://github.com/delneg/noteshrinker-django)
