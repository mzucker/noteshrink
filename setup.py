#!/usr/bin/env python3
import os
import sys
from setuptools import setup

if sys.argv[-1] == "publish":
    if os.system("pip3 freeze | grep wheel"):
        print("wheel not installed.\nUse `pip install wheel`.\nExiting.")
        sys.exit()
    os.system("python3 setup.py sdist upload")
    os.system("python3 setup.py bdist_wheel upload")
    print("You probably want to also tag the version now:")
    print("  git tag -a VERSION -m 'version VERSION'")
    print("  git push --tags")
    sys.exit()

setup(
    name="noteshrink",
    version="0.1.0",
    author="Matt Zucker",
    description="Convert scans of handwritten notes to beautiful, compact PDFs",
    url="https://github.com/mzucker/noteshrink",
    py_modules=["noteshrink"],
    install_requires=[
        "numpy",
        "scipy",
        "pillow",
    ],
    entry_points="""
        [console_scripts]
        noteshrink=noteshrink:main
    """,
)
