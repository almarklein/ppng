# Copyright (c) 2016-2025, Almar Klein
# This module is distributed under the terms of the MIT License.
#
# This module is small enough so that it can be included in a larger project.
# Or you can make ppng a (light) dependency.

"""
Pure python module to handle for reading and writing png files.

Supports:
- Greyscale, greyscale-plus-alpha, RGB, RGBA.
- 8 bit and 16 bit.
- Chunked writing.
- Animated images.
- Volumetric images.

Does not support:
- interlacing.
- paletted images.

"""

import zlib
import struct
import pathlib
import logging
from typing import Union

__all__ = ["PngWriter", "write_apng", "write_png"]

__version__ = "1.0.0"
version_info = tuple(int(i) if i.isnumeric() else i for i in __version__.split("."))

logger = logging.getLogger("ppng")


# ----- utils


COLOR_TYPES = {
    # "p": (0b011, 1, (1, 2, 4, 8)),
    "l": (0b000, 1, (1, 2, 4, 8, 16)),
    "la": (0b100, 2, (8, 16)),
    "rgb": (0b010, 3, (8, 16)),
    "rgba": (0b110, 4, (8, 16)),
}
N_CHANNELS_TO_FORMAT = ["zero", "l", "la", "rgb", "rgba"]


def _get_im_props_from_memoryview(mem):
    width, height = mem.shape[1], mem.shape[0]
    if len(mem.shape) == 2:
        format = "l"
    elif len(mem.shape) == 3:
        format = N_CHANNELS_TO_FORMAT[mem.shape[2]]
    else:
        raise ValueError("Unexpected data shape {mem.shape}")
    return width, height, format


# ----- writing


def write_png(file, image, *, compression=9):
    """Write an image to a png file.

    Parameters:
        file : str | pathlib.Path | file-like
            The file to write to. Can be a filename or anything that has a ``write()`` method.
        image : array-like
            The image to write. Can be anything that can be cast to a memoryview (e.g. a numpy array).
        compression : int
            The zlib compression level to use, with 0 no compression, and 9 highest compression. Default 9.
    """
    mem = memoryview(image)
    width, height, format = _get_im_props_from_memoryview(mem)
    with PngWriter(file, width, height, format, compression=compression) as writer:
        writer.write(mem)


def write_apng(file, images, compression=9):
    """ "Write images to an apng file (animated png)."""

    frame_count = len(images)
    image_iter = iter(images)
    first_image = next(image_iter)

    width, height, format = _get_im_props_from_memoryview(memoryview(first_image))

    with PngWriter(
        file,
        width,
        height,
        format,
        png_mode="apng",
        frame_count=frame_count,
        compression=compression,
    ) as writer:
        writer.write_animation_frame(first_image)
        for image in image_iter:
            writer.write_animation_frame(image)


class PngWriter:
    """Object that supports streamed writing to a png file.

    For details, see the docs of write_png.

    Usage:

        with PngWriter(file, width, height, format) as writer:
            writer.write(part)
            ...
            writer.write(part)
    """

    def __init__(
        self,
        file,
        width,
        height,
        format,
        *,
        png_mode="png",
        frame_count=1,
        compression=9,
    ):
        # Get file handle
        self.file = file
        if isinstance(self.file, (str, pathlib.Path)):
            self.fh = open(self.file, "wb")
        elif hasattr(self.file, "write"):
            self.fh = self.file
        else:
            raise RuntimeError(f"Cannot write to {self.file}")

        self.width = int(width)
        self.height = int(height)
        self.format = str(format).lower()
        if self.format not in COLOR_TYPES:
            raise ValueError("Unknown/unsupported color format: '{format}'")

        # Configuration
        self._compression = int(compression)
        self._png_mode = str(png_mode)
        self._frame_count = int(frame_count)  # for apng

        # State
        self._animation_frames_written = 0
        self._apng_sequence_number = 0  # explicitly order the apng chunks
        self._idat_written = False

    def __enter__(self):
        # Write signature and header chunk
        self._write_signature()
        self._write_chunk_ihdr()
        if self._png_mode == "apng":
            self._write_chunk_actl()
        return self

    def __exit__(self, value, type, tb):
        # todo: check that all is written
        self._write_chunk_iend()
        if self.fh is not self.file:
            self.fh.close()
        self.fh = None

        # Sanity checks
        if self._png_mode == "apng":
            if self._animation_frames_written != self._frame_count:
                logger.warning(
                    f"PNGWriter wrote has frame_count {self._frame_count} but wrote {self._animation_frames_written} animation frames."
                )

    def write(self, image):
        if self.fh is None:
            raise RuntimeError("Attempt to write to PngWriter after it has finished.")
        mem = memoryview(image)
        # todo: check
        # todo: split in smaller chunks automatically
        self._idat_written = True
        self._write_chunk_idat(mem)

    def write_animation_frame(
        self,
        image,
        *,
        delay=0.1,
        x_offset=0,
        y_offset=0,
        dispose_op=0,
        blend_op=0,
    ):
        if self._png_mode != "apng":
            raise RuntimeError("Not an apng file")
        if self.fh is None:
            raise RuntimeError("Attempt to write to PngWriter after it has finished.")
        mem = memoryview(image)

        width, height = mem.shape[1], mem.shape[0]
        x_offset, y_offset = int(x_offset), int(y_offset)
        if (
            x_offset < 0
            or y_offset < 0
            or x_offset + width > self.width
            or y_offset + height > self.height
        ):
            raise RuntimeError(
                "Animation frame must be contained within the reference image."
            )

        self._animation_frames_written += 1

        if isinstance(delay, tuple):
            delay_num, delay_den = int(delay[0]), int(delay[1])
        else:
            delay = float(delay)
            delay_num, delay_den = (int(delay * 1000), 1000)

        dispose_op = int(dispose_op)
        dispose_op = dispose_op if dispose_op in (0, 1, 2) else 0

        blend_op = int(blend_op)
        blend_op = blend_op if blend_op in (0, 1) else 0

        # todo: implement sub-rectangle detection

        self._write_chunk_fctl(
            width,
            height,
            x_offset,
            y_offset,
            delay_num,
            delay_den,
            dispose_op,
            blend_op,
        )

        # First frame can be the static image, but the static image can also be independent
        if not self._idat_written:
            self._idat_written = True
            self._write_chunk_idat(mem)
        else:
            self._write_chunk_fdat(mem)

    def _write_signature(self):
        self.fh.write(b"\x89PNG\x0d\x0a\x1a\x0a")  # signature

    def _write_chunk(self, name: str, datas: Union[bytes, list]):
        print("writing chubnk", name)
        if isinstance(datas, bytes):
            datas = [datas]
        name = name.encode("ASCII")
        # Get checksum and byte count
        checksum = zlib.crc32(name)
        nbytes = 0
        for bb in datas:
            nbytes += len(bb)
            checksum = zlib.crc32(bb, checksum)
        # Write
        self.fh.write(struct.pack(">I", nbytes))
        self.fh.write(name)
        for bb in datas:
            self.fh.write(bb)
        self.fh.write(checksum.to_bytes(4, "big"))

    def _write_chunk_ihdr(self):
        # The image header
        depth = 8
        ctyp = COLOR_TYPES[self.format][0]
        data = struct.pack(">IIBBBBB", self.width, self.height, depth, ctyp, 0, 0, 0)
        self._write_chunk("IHDR", data)

    def _write_chunk_idat(self, mem: memoryview):
        # The image data
        self._write_chunk("IDAT", self._compress_image_data(mem))

    def _compress_image_data(self, mem):
        datas = []
        c = zlib.compressobj(self._compression, 8, 15, 9)
        for i in range(mem.shape[0]):
            scanline = mem[i : i + 1]
            filter_flag = b"\x00"  # no filter
            bb = c.compress(filter_flag)
            if bb:
                datas.append(bb)
            bb = c.compress(scanline)
            if bb:
                datas.append(bb)
        datas.append(c.flush())
        return datas

    def _write_chunk_iend(self):
        # The image trailer
        self._write_chunk("IEND", b"")

    def _write_chunk_actl(self):
        # The animation chunk control
        # Identifies that this is an animated png. Note that the frame count must be exact.
        num_plays = 0  # loop indefinitely
        self._write_chunk("acTL", struct.pack(">II", self._frame_count, num_plays))

    def _write_chunk_fctl(self, *args):
        # The frame control chunk
        # args = width, height, x_offset, y_offset, delay_num, delay_den, dispose_op, blend_op
        sequence_number = self._apng_sequence_number
        self._apng_sequence_number += 1
        data = struct.pack(">IIIIIHHBB", sequence_number, *args)
        self._write_chunk("fcTL", data)

    def _write_chunk_fdat(self, mem: memoryview):
        # The frame data chunk
        sequence_number = self._apng_sequence_number
        self._apng_sequence_number += 1
        datas = self._compress_image_data(mem)
        datas.insert(0, struct.pack(">I", sequence_number))
        self._write_chunk("fdAT", datas)


# ----- reading


def read_png(f, return_ndarray=False):
    """
    Read a png image. This is a simple implementation; can only read
    PNG's that are not interlaced, have a bit depth of 8, and are either
    RGB or RGBA.

    Parameters:
        f (file-object, bytes): the source to read the png data from.
        return_ndarray (bool): whether to return the result as a numpy array.
            Default False. If False, returns ``(pixel_array, shape)``,
            with ``pixel_array`` a bytearray object and shape being
            ``(H, W, 3)`` or ``(H, W, 4)``, for RGB and RGBA, respectively.
    """
    # http://en.wikipedia.org/wiki/Portable_Network_Graphics
    # http://www.libpng.org/pub/png/spec/1.2/PNG-Chunks.html

    asint_map = {1: ">B", 2: ">H", 4: ">I"}
    asint = lambda x: struct.unpack(asint_map[len(x)], x)[0]

    # Get bytes
    if isinstance(f, (bytes, bytearray)):
        bb = f
    elif hasattr(f, "read"):
        bb = f.read()
    else:
        raise TypeError("read_png() needs file object or bytes, not %r" % f)

    # Read header
    if not (bb[0:1] == b"\x89" and bb[1:4] == b"PNG"):
        raise RuntimeError(
            "Image data does not appear to have a PNG " "header: %r" % bb[:10]
        )
    chunk_pointer = 8

    # Read first chunk
    chunk1 = bb[chunk_pointer:]
    chunk_length = asint(chunk1[0:4])
    chunk_pointer += 12 + chunk_length  # size, type, crc, data
    if not (chunk1[4:8] == b"IHDR" and chunk_length == 13):  # noqa
        raise RuntimeError("Unable to read PNG data, maybe its corrupt?")

    # Extract info
    width = asint(chunk1[8:12])
    height = asint(chunk1[12:16])
    bit_depth = asint(chunk1[16:17])
    color_type = asint(chunk1[17:18])
    compression_method = asint(chunk1[18:19])
    filter_method = asint(chunk1[19:20])
    interlace_method = asint(chunk1[20:21])
    bytes_per_pixel = 3 + (color_type == 6)

    # Check if we can do this ....
    if bit_depth != 8:
        raise RuntimeError("Can only deal with bit-depth of 8.")
    if color_type not in [2, 6]:  # RGB, RGBA
        raise RuntimeError("Can only deal with RGB or RGBA.")
    if interlace_method != 0:
        raise RuntimeError("Can only deal with non-interlaced.")
    if filter_method != 0:
        raise RuntimeError("Can only deal with unfiltered data.")
    if compression_method != 0:
        # this should be the case for any PNG
        raise RuntimeError("Expected PNG compression param to be 0.")

    # If this is the case ... extract pixel info
    lines, prev = [], None
    while True:
        chunk = bb[chunk_pointer:]
        if not chunk:
            break
        chunk_length = asint(chunk[0:4])
        chunk_pointer += 12 + chunk_length
        if chunk[4:8] == b"IEND":
            break
        elif chunk[4:8] == b"IDAT":  # Pixel data
            # Decompress and unfilter
            pixels_compressed = chunk[8 : 8 + chunk_length]
            pixels_raw = zlib.decompress(pixels_compressed)
            s = width * bytes_per_pixel + 1  # stride
            # print(pixels_raw[0::s])  # show filters in use
            for i in range(height):
                prev = _png_scanline(pixels_raw[i * s : i * s + s], prev=prev)
                lines.append(prev)

    # Combine scanlines from all chunks
    im = bytearray(sum([len(line) for line in lines]))
    i = 0
    line_len = width * bytes_per_pixel
    for line in lines:
        if not len(line) == line_len:  # noqa
            raise RuntimeError("Line length mismatch while reading png.")
        im[i : i + line_len] = line
        i += line_len

    shape = height, width, bytes_per_pixel

    # Done
    if return_ndarray:
        import numpy as np

        return np.frombuffer(im, "uint8").reshape(shape)
    else:
        return im, shape


def _png_scanline(line_bytes, fu=4, prev=None):
    """Scanline unfiltering, taken from png.py"""
    filter = ord(line_bytes[0:1])
    line1 = bytearray(line_bytes[1:])  # copy so that indexing yields ints
    line2 = bytearray(line_bytes[1:])  # output line

    if filter == 0:
        pass  # No filter
    elif filter == 1:
        # sub
        ai = 0
        for i in range(fu, len(line2)):
            x = line1[i]
            a = line2[ai]
            line2[i] = (x + a) & 0xFF
            ai += 1
    elif filter == 2:
        # up
        for i in range(len(line2)):
            x = line1[i]
            b = prev[i]
            line2[i] = (x + b) & 0xFF
    elif filter == 3:
        # average
        ai = -fu
        for i in range(len(line2)):
            x = line1[i]
            if ai < 0:
                a = 0
            else:
                a = line2[ai]
            b = prev[i]
            line2[i] = (x + ((a + b) >> 1)) & 0xFF
            ai += 1
    elif filter == 4:
        # paeth
        ai = -fu  # Also used for ci.
        for i in range(len(line2)):
            x = line1[i]
            if ai < 0:
                a = c = 0
            else:
                a = line2[ai]
                c = prev[ai]
            b = prev[i]
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa <= pb and pa <= pc:
                pr = a
            elif pb <= pc:
                pr = b
            else:
                pr = c
            line2[i] = (x + pr) & 0xFF
            ai += 1
    else:
        raise RuntimeError("Invalid filter %r" % filter)
    return line2
