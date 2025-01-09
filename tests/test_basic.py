import pathlib

import ppng
import numpy as np
import PIL.Image

suite_dir = pathlib.Path(__file__).parent / "png_test_suite"


def test_basic_gray():
    filename = suite_dir / "basn0g08.png"

    im = ppng.read_png(filename)
    im = np.asarray(im)

    assert im.shape == (32, 32)

    ref = np.asarray(PIL.Image.open(filename))

    assert im.shape == ref.shape
    assert im.dtype == ref.dtype
    assert np.all(im == ref)


def test_basic_graya():
    filename = suite_dir / "basn4a08.png"

    im = ppng.read_png(filename)
    im = np.asarray(im)

    assert im.shape == (32, 32, 2)

    ref = np.asarray(PIL.Image.open(filename))

    assert im.shape == ref.shape
    assert im.dtype == ref.dtype
    assert np.all(im == ref)


def test_basic_rgb():
    filename = suite_dir / "basn2c08.png"

    im = ppng.read_png(filename)
    im = np.asarray(im)

    assert im.shape == (32, 32, 3)

    ref = np.asarray(PIL.Image.open(filename))

    assert im.shape == ref.shape
    assert im.dtype == ref.dtype
    assert np.all(im == ref)


def test_basic_rgba():
    filename = suite_dir / "basn6a08.png"

    im = ppng.read_png(filename)
    im = np.asarray(im)

    assert im.shape == (32, 32, 4)

    ref = np.asarray(PIL.Image.open(filename))

    assert im.shape == ref.shape
    assert im.dtype == ref.dtype
    assert np.all(im == ref)


if __name__ == "__main__":
    test_basic_gray()
    test_basic_graya()
    test_basic_rgb()
    test_basic_rgba()
