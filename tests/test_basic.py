import os
import sys
import pathlib
import tempfile
import subprocess

import ppng
import numpy as np
import PIL.Image


suite_dir = pathlib.Path(__file__).parent / "png_test_suite"
temp_dir = pathlib.Path(tempfile.gettempdir())


# Is pngcheck installed on the system?
try:
    subprocess.run(["pngcheck", "notafile"], capture_output=True)
except FileNotFoundError:
    HAS_PNGCHECK = False
else:
    HAS_PNGCHECK = False


def pillow_read(filename):
    return np.asarray(PIL.Image.open(filename))


def pngcheck(filename):
    if not HAS_PNGCHECK:
        return  # skip
    p = subprocess.run(["pngcheck", filename], capture_output=True)
    out = p.stdout.decode()
    if out.startswith("OK"):
        return False
    else:
        return out


# -----


def test_has_pngcheck():
    # Make sure pngcheck is installed on Linux CI
    if os.getenv("CI") and sys.platform.startswith("linux"):
        assert HAS_PNGCHECK


def test_basic_gray():
    # Read with ppng, check with pillow

    filename = suite_dir / "basn0g08.png"

    im1 = ppng.read_png(filename)
    im1 = np.asarray(im1)

    assert im1.shape == (32, 32)

    im2 = pillow_read(filename)

    assert im1.shape == im2.shape
    assert im1.dtype == im2.dtype
    assert np.all(im1 == im2)

    # Write with ppng

    filename = temp_dir / filename.name

    ppng.write_png(filename, im1)

    assert not pngcheck(filename)

    im3 = np.asarray(ppng.read_png(filename))

    assert im3.shape == im1.shape
    assert im3.dtype == im1.dtype
    assert np.all(im3 == im1)

    im4 = pillow_read(filename)

    assert im4.shape == im1.shape
    assert im4.dtype == im1.dtype
    assert np.all(im4 == im1)


def test_basic_graya():
    # Read with ppng, check with pillow

    filename = suite_dir / "basn4a08.png"

    im1 = ppng.read_png(filename)
    im1 = np.asarray(im1)

    assert im1.shape == (32, 32, 2)

    im2 = pillow_read(filename)

    assert im1.shape == im2.shape
    assert im1.dtype == im2.dtype
    assert np.all(im1 == im2)

    # Write with ppng

    filename = temp_dir / filename.name

    ppng.write_png(filename, im1)

    assert not pngcheck(filename)

    im3 = np.asarray(ppng.read_png(filename))

    assert im3.shape == im1.shape
    assert im3.dtype == im1.dtype
    assert np.all(im3 == im1)

    im4 = pillow_read(filename)

    assert im4.shape == im1.shape
    assert im4.dtype == im1.dtype
    assert np.all(im4 == im1)


def test_basic_rgb():
    # Read with ppng, check with pillow

    filename = suite_dir / "basn2c08.png"

    im1 = ppng.read_png(filename)
    im1 = np.asarray(im1)

    assert im1.shape == (32, 32, 3)

    im2 = pillow_read(filename)

    assert im1.shape == im2.shape
    assert im1.dtype == im2.dtype
    assert np.all(im1 == im2)

    # Write with ppng

    filename = temp_dir / filename.name

    ppng.write_png(filename, im1)

    assert not pngcheck(filename)

    im3 = np.asarray(ppng.read_png(filename))

    assert im3.shape == im1.shape
    assert im3.dtype == im1.dtype
    assert np.all(im3 == im1)

    im4 = pillow_read(filename)

    assert im4.shape == im1.shape
    assert im4.dtype == im1.dtype
    assert np.all(im4 == im1)


def test_basic_rgba():
    # Read with ppng, check with pillow

    filename = suite_dir / "basn6a08.png"

    im1 = ppng.read_png(filename)
    im1 = np.asarray(im1)

    assert im1.shape == (32, 32, 4)

    im2 = pillow_read(filename)

    assert im1.shape == im2.shape
    assert im1.dtype == im2.dtype
    assert np.all(im1 == im2)

    # Write with ppng

    filename = temp_dir / filename.name

    ppng.write_png(filename, im1)

    assert not pngcheck(filename)

    im3 = np.asarray(ppng.read_png(filename))

    assert im3.shape == im1.shape
    assert im3.dtype == im1.dtype
    assert np.all(im3 == im1)

    im4 = pillow_read(filename)

    assert im4.shape == im1.shape
    assert im4.dtype == im1.dtype
    assert np.all(im4 == im1)


if __name__ == "__main__":
    test_basic_gray()
    test_basic_graya()
    test_basic_rgb()
    test_basic_rgba()
