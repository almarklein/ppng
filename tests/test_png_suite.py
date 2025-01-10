"""
Run tests based on the "officia" png test suite from
http://www.schaik.com/pngsuite/pngsuite.html
"""

import pathlib

import ppng
import pytest


suite_dir = pathlib.Path(__file__).parent / "png_test_suite"

all_images = sorted(x.name for x in suite_dir.glob("*.png") if x.name != "PngSuite.png")


def name2props(name):
    name = name.split(".")[0]
    if name.startswith("exif"):
        name = "exifn" + name[4:]
    bit_depth = int(name[-2:].lstrip("0"))
    color_type = name[-4:-2]
    color_type = {
        "0g": "gray",
        "2c": "rgb",
        "3p": "paletted",
        "4a": "graya",
        "6a": "rgba",
    }[color_type]
    interlaced = {"n": "non-interlaced", "i": "interlaced"}[name[-5]]

    if name.startswith("bas"):
        category, param = "basic", ""
    elif name.startswith("bg"):
        category, param = "background", name[2:-5]
    elif name.startswith("oi"):
        category, param = "chunk-order", name[2:-5]

    elif name.startswith("s"):
        category, param = "odd-sizes", name[1:-5]
    elif name.startswith("t"):
        category, param = "transparency", name[1:-5]
    elif name.startswith("g"):
        category, param = "gamma", name[1:-5]
    elif name.startswith("f"):
        category, param = "filtering", name[1:-5]
    elif name.startswith("p"):
        category, param = "additional", name[1:-5]
    elif name.startswith("c"):
        category, param = "ancillary", name[1:-5]
    elif name.startswith("z"):
        category, param = "zlib", name[1:-5]
    elif name.startswith("x"):
        category, param = "corrupted", name[1:-5]
    else:
        category, param = "unknown", name[:-5]

    return category, param, interlaced, color_type, bit_depth


def name2summary(name):
    category, param, interlaced, color_type, bit_depth = name2props(name)
    return f"{name} {category} {param} {interlaced} {color_type} {bit_depth}bits"


# ---


@pytest.mark.parametrize("name", all_images, ids=name2summary)
def test_image_from_suite(name):
    filename = suite_dir / name
    category, param, interlaced, color_type, bit_depth = name2props(name)

    if category == "corrupted":
        pytest.skip()
        # todo: test analysing corrupt files
        # todo: test fail with appropriate error message
    if interlaced == "interlaced":
        pytest.skip()
        # todo: support for interlaced images?
        # todo: if not, fail with appropriate error message
    if bit_depth != 8:
        pytest.skip()
        # todo support for other bit depths

    im = ppng.read_png(filename)
    assert isinstance(im, memoryview)
    assert len(im.shape) == 2 or len(im.shape) == 3

    # todo: also compare the result to a reference

    if category == "gamma":
        # http://www.schaik.com/pngsuite/pngsuite.html#gamma
        # todo: do some extra checks for gamma
        pass

    if category == "ancillary":
        # http://www.schaik.com/pngsuite/pngsuite.html#ancillary"
        # todo: som extra checks for ancillary data
        pass


if __name__ == "__main__":
    # Allow running as a script
    for name in all_images:
        print(name2summary(name))
        try:
            test_image_from_suite(name)
        except BaseException as err:
            if err.__class__.__name__ == "Skipped":
                print("Skip")
                continue
            raise
        else:
            print("Done")
