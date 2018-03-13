noteshrink
==========

Convert scans of handwritten notes to beautiful, compact PDFs -- see full writeup at <https://mzucker.github.io/2016/09/20/noteshrink.html>
Original work by Matt Zucker <https://github.com/mzucker/noteshrink>

## Requirements

 - Python 3

Python libraries: (use requirements.txt)
 - NumPy 1.10 or later
 - SciPy 1.0.0 or later
 - Image module from Pillow 5.0.0 or later

Also:
 - ImageMagick (for "convert")
   - on macOS: `brew install imagemagick` (requires Homebrew)
   - on Debian/Ubuntu: `sudo apt-get install imagemagick`

## Installation

Install python dependencies: (Don't forget virtualenv)
```
make install
```
or 
```
pip3 install -r requirements.txt
```

Install imagemagick:
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
  - [Noteshrink Docker Image](https://hub.docker.com/r/johnpaulada/noteshrink/)
