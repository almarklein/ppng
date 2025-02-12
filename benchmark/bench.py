"""
Benchmarks with png and other formats, to measure performance of compression
and write speeds.

This code defines 12 natural images and 12 artificial images to distinguish
between these cases.

"""

import os
import time
import pathlib

import numpy as np
import ppng
import PIL
import simplejpeg
import matplotlib.pyplot as plt


plt.ion()

this_dir = pathlib.Path(__file__).parent
out_dir = this_dir / "out"


natural_images = [
    "astronaut.png",  # (512, 512, 3)
    "camera.png",  # (512, 512)
    "chelsea.png",  # (300, 451, 3)
    "wikkie.png",  # (512, 512, 3)
    "text.png",  # (172, 448)
    "moon.png",  # (512, 512)
    "coffee.png",  # (400, 600, 3)
    "coins.png",  # (303, 384)
    "horse.png",  # (328, 400, 4)
    "immunohistochemistry.png",  # (512, 512, 3)
    "bricks.png",  # (512, 512, 3)
    "wood.png",  # (512, 512, 3)
]

artificial_images = [
    "blend_bg.png",  # (600, 600, 4)
    "points_size.png",  # (600, 800, 4)
    "blend_additive.png",  # (480, 640, 4)
    "blend_ordered2.png",  # (600, 600, 4)
    "text_align.png",  # (600, 800, 4)
    "color.png",  # (480, 640, 4)
    "image2.png",  # (480, 640, 4)
    "blend_dither.png",  # (600, 600, 4)
    "normals_sides.png",  # (480, 640, 4)
    "points_markers.png",  # (1000, 1200, 4)
    "grid1.png",  # (480, 640, 4)
    "volume.png",  # (480, 640, 4)
]


def load_natural_images():
    for fname in natural_images:
        im = ppng.read_png(this_dir / "natural_images" / fname)
        yield fname, np.asarray(im)


def load_artificial_images():
    for fname in artificial_images:
        im = ppng.read_png(this_dir / "artificial_images" / fname)
        yield fname, np.asarray(im)


class TestCase:
    """Class that can be subclasses to create a testcase."""

    name = "TestCase"
    ext = ".unknown"
    xlabel = "xlabel"
    write_bounds = 0, 100
    read_bounds = 0, 100
    mem_bounds1 = 0, 100
    mem_bounds2 = 0, 100

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_name(self):
        return self.name

    def iterx(self):
        raise NotImplementedError()

    def write(self, x, filename, im):
        raise NotImplementedError()

    def read(self, filename):
        raise NotImplementedError()

    def measure(self):
        self.xvalues = np.array(list(self.iterx()))
        self.read_sets1, self.write_sets1, self.size_sets1 = self._measure(
            load_natural_images
        )
        self.read_sets2, self.write_sets2, self.size_sets2 = self._measure(
            load_artificial_images
        )

    def _measure(self, image_gen):
        read_sets = []
        write_sets = []
        size_sets = []

        for fname, im in image_gen():
            if self.ext == ".jpg" and im.ndim == 3 and im.shape[2] == 4:
                im = im[:, :, :3].copy()
            fname = fname.split(".")[0] + self.ext
            print(self.get_name(), fname, im.shape)
            filename = out_dir / fname
            read_times = []
            write_times = []
            sizes = []

            for x in self.iterx():
                imc = im.copy()

                t0 = time.perf_counter()
                self.write(x, filename, imc)
                t1 = time.perf_counter()
                im2 = self.read(filename)
                t2 = time.perf_counter()

                im2 = np.asarray(im2)
                if self.ext == ".jpg" and im.ndim == 2:
                    assert im2.shape[:2] == im.shape[:2]
                elif self.ext == ".webp":  # sigh
                    assert im2.shape[:2] == im.shape[:2]
                else:
                    assert im2.shape == im.shape
                if self.ext in (".png",):
                    assert np.all(im2 == im)

                read_times.append(t1 - t0)
                write_times.append(t2 - t1)
                sizes.append(filename.stat().st_size)
                os.unlink(filename)

            read_times = np.array(read_times, float)
            write_times = np.array(write_times, float)
            sizes = np.array(sizes, float)
            sizes = (sizes / im.nbytes) * 100
            read_sets.append(read_times * 100)
            write_sets.append(write_times * 100)
            size_sets.append(sizes)

        return read_sets, write_sets, size_sets


def make_plots(nrows, row, case):
    """Make plots for the given case, on one row of the current figure."""
    ncols = 4
    xvalues = case.xvalues  # raise Attribute error if not measurement has been done

    plt.subplot(nrows, ncols, row * ncols + 1)
    plt.title(case.get_name())
    plt.grid()
    plt.xlabel(case.xlabel)
    plt.ylabel("Write time (ms)")
    for times in case.read_sets1:
        plt.plot(xvalues, times, color="green", alpha=0.2, linewidth=1)
    plt.plot(xvalues, np.mean(case.read_sets1, axis=0), color="green", linewidth=4)
    for times in case.read_sets2:
        plt.plot(xvalues, times, color="purple", alpha=0.2, linewidth=1)
    plt.plot(xvalues, np.mean(case.read_sets2, axis=0), color="purple", linewidth=4)
    plt.ylim(*case.write_bounds)

    plt.subplot(nrows, ncols, row * ncols + 2)
    plt.grid()
    plt.xlabel(case.xlabel)
    plt.ylabel("Read time (ms)")
    for times in case.write_sets1:
        plt.plot(xvalues, times, color="green", alpha=0.2, linewidth=1)
    plt.plot(xvalues, np.mean(case.write_sets1, axis=0), color="green", linewidth=4)
    for times in case.write_sets2:
        plt.plot(xvalues, times, color="purple", alpha=0.2, linewidth=1)
    plt.plot(xvalues, np.mean(case.write_sets2, axis=0), color="purple", linewidth=4)
    plt.ylim(*case.read_bounds)

    plt.subplot(nrows, ncols, row * ncols + 3)
    plt.xlabel(case.xlabel)
    plt.ylabel("Size (%)")
    plt.grid()
    for sizes in case.size_sets1:
        plt.plot(xvalues, sizes, color="green", alpha=0.2, linewidth=1)
    plt.plot(xvalues, np.mean(case.size_sets1, axis=0), color="green", linewidth=4)
    plt.ylim(*case.mem_bounds1)

    plt.subplot(nrows, ncols, row * ncols + 4)
    plt.xlabel(case.xlabel)
    plt.ylabel("Size (%)")
    plt.grid()
    for sizes in case.size_sets2:
        plt.plot(xvalues, sizes, color="purple", alpha=0.2, linewidth=1)
    plt.plot(xvalues, np.mean(case.size_sets2, axis=0), color="purple", linewidth=4)
    plt.ylim(*case.mem_bounds2)


# %%%%% Test cases


class TestCasePng(TestCase):
    name = "png filter 0"
    filter = 0
    ext = ".png"
    xlabel = "compression"
    write_bounds = 0, 5
    read_bounds = 0, 1
    mem_bounds1 = 0, 80
    mem_bounds2 = 0, 7

    def get_name(self):
        filtername = ["Unfiltered", "Sub", "Up", "Average", "Paeth"][self.filter]
        return f"ppng filter {self.filter} ({filtername})"

    def iterx(self):
        for i in range(10):
            yield i

    def write(self, x, filename, im):
        ppng.write_png(filename, im, filter=self.filter, compression=x)

    def read(self, filename):
        return ppng.read_png(filename)


class TestCasePillowPng(TestCase):
    name = "png Pillow"
    ext = ".png"
    xlabel = "compression"
    write_bounds = 0, 5
    read_bounds = 0, 1
    mem_bounds1 = 0, 80
    mem_bounds2 = 0, 7

    def iterx(self):
        for i in range(10):
            yield i

    def write(self, x, filename, im):
        pim = PIL.Image.fromarray(im)
        pim.save(filename, compress_level=x, optimize=False)

    def read(self, filename):
        image = PIL.Image.open(filename)
        return np.asarray(image)


class TestCasePillowJpg(TestCase):
    name = "jpg Pillow"
    ext = ".jpg"
    xlabel = "quality"
    write_bounds = 0, 0.3
    read_bounds = 0, 0.3
    mem_bounds1 = 0, 20
    mem_bounds2 = 0, 3

    def iterx(self):
        for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            yield i

    def write(self, x, filename, im):
        pim = PIL.Image.fromarray(im)
        pim.save(filename, quality=x, optimize=False)

    def read(self, filename):
        image = PIL.Image.open(filename)
        return np.asarray(image)


class TestCaseSimpleJpg(TestCase):
    name = "jpg turbo"
    ext = ".jpg"
    xlabel = "quality"
    write_bounds = 0, 0.3
    read_bounds = 0, 0.3
    mem_bounds1 = 0, 20
    mem_bounds2 = 0, 3

    def iterx(self):
        for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            yield i

    def write(self, x, filename, im):
        if im.ndim == 2:
            colorspace = "GRAY"
            im.shape = (*im.shape, 1)
        elif im.shape[2] == 1:
            colorspace = "GRAY"
        elif im.shape[2] == 3:
            colorspace = "RGB"
        elif im.shape[2] == 4:
            colorspace = "RGBA"
        else:
            raise NotImplementedError()

        bb = simplejpeg.encode_jpeg(im, x, colorspace, "444")
        with open(filename, "wb") as f:
            f.write(bb)

    def read(self, filename):
        with open(filename, "rb") as f:
            bb = f.read()
        image = simplejpeg.decode_jpeg(bb)
        return np.asarray(image)


class TestCaseAvif(TestCase):
    # There is https://github.com/Julian/avif, but it only has a decoder,
    # and it also *very* minimal.
    name = "Avif"


class TestCaseWebp(TestCase):
    name = "WebP"
    ext = ".webp"
    xlabel = "quality"

    def iterx(self):
        for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            yield i

    def write(self, x, filename, im):
        pim = PIL.Image.fromarray(im)
        pim.save(filename, quality=x, lossless=self.lossless)

    def read(self, filename):
        image = PIL.Image.open(filename)
        return np.asarray(image)


class TestCaseWebpLossless(TestCaseWebp):
    name = "WebP lossless"
    lossless = True
    write_bounds = 0, 20
    read_bounds = 0, 1
    mem_bounds1 = 0, 80
    mem_bounds2 = 0, 7


class TestCaseWebpLossy(TestCaseWebp):
    name = "WebP lossy"
    lossless = False
    write_bounds = 0, 5
    read_bounds = 0, 1
    mem_bounds1 = 0, 20
    mem_bounds2 = 0, 3


# %%%%% Compare PNG filters

testcases = [
    TestCasePng(filter=0),
    TestCasePng(filter=1),
    TestCasePng(filter=2),
    TestCasePng(filter=3, read_bounds=(0, 40)),
    TestCasePng(filter=4, read_bounds=(0, 40)),
]

for case in testcases:
    case.measure()

fig = plt.figure(1)
fig.clear()
fig.set_size_inches(12, len(testcases) * 3)

for i, case in enumerate(testcases):
    make_plots(len(testcases), i, case)

fig.tight_layout()


# %%%%% Compare jpeg

testcases = [
    TestCasePng(filter=2, write_bounds=(0, 3)),
    TestCaseSimpleJpg(),
]

for case in testcases:
    case.measure()

fig = plt.figure(2)
fig.clear()
fig.set_size_inches(12, len(testcases) * 3)

for i, case in enumerate(testcases):
    make_plots(len(testcases), i, case)

fig.tight_layout()


# %%%%% Compare Pillow

testcases = [
    TestCasePillowPng(),
    TestCasePillowJpg(),
]

for case in testcases:
    case.measure()

fig = plt.figure(3)
fig.clear()
fig.set_size_inches(12, len(testcases) * 3)

for i, case in enumerate(testcases):
    make_plots(len(testcases), i, case)

fig.tight_layout()


# %%%%%Compare Webp

testcases = [
    TestCaseWebpLossless(),
    TestCaseWebpLossy(),
]

for case in testcases:
    case.measure()

fig = plt.figure(3)
fig.clear()
fig.set_size_inches(12, len(testcases) * 3)

for i, case in enumerate(testcases):
    make_plots(len(testcases), i, case)

fig.tight_layout()
