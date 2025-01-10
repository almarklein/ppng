# Copyright (c) 2016-2025, Almar Klein
# This module is distributed under the terms of the MIT License.
#
# This module is small enough so that it can be included in a larger project.
# Or you can make ppng a (light) dependency.

"""
Pure python module for writing png images.

Supports:
- Grayscale, grayscale+alpha, RGB, RGBA.
- 8 bit and 16 bit.
- Chunked writing.
- Animated images.
- Volumetric images.

Does not support:
- interlacing.
- paletted images.

"""

__all__ = ["PngWriter", "write_apng", "write_png"]


import zlib
import struct
import pathlib
import logging
import itertools
from typing import Union


__version__ = "1.0.0"
version_info = tuple(int(i) if i.isnumeric() else i for i in __version__.split("."))


logger = logging.getLogger("ppng")

# todo: when to warn, when to error?

COLOR_TYPES = {
    # flag, pixel-size, format, bitdepths
    (0b011, 1, "p", (1, 2, 4, 8)),
    (0b000, 1, "l", (1, 2, 4, 8, 16)),
    (0b100, 2, "la", (8, 16)),
    (0b010, 3, "rgb", (8, 16)),
    (0b110, 4, "rgba", (8, 16)),
}

FORMAT_TO_COLOR_FLAG = {ct[2]: ct[0] for ct in COLOR_TYPES}
FORMAT_TO_PSIZE = {ct[2]: ct[1] for ct in COLOR_TYPES}
N_CHANNELS_TO_FORMAT = {ct[1]: ct[2] for ct in COLOR_TYPES}
COLOR_FLAG_TO_FORMAT = {ct[0]: ct[2] for ct in COLOR_TYPES}


def get_im_props_from_memoryview(mem):
    width, height = mem.shape[1], mem.shape[0]
    if len(mem.shape) == 2:
        format = "l"
    elif len(mem.shape) == 3:
        format = N_CHANNELS_TO_FORMAT[mem.shape[2]]
    else:
        raise ValueError("Unexpected data shape {mem.shape}")
    return width, height, format


def write_png(file, image, *, compression=9):
    """Write an image to a png file.

    Parameters:
        file : str | pathlib.Path | file-like
            The file to write to. Can be a filename or anything that has a
            ``write()`` method.
        image : array-like
            The image to write. Can be anything that can be cast to a memoryview
            (e.g. a numpy array).
        compression : int
            The zlib compression level to use, with 0 no compression, and 9 highest
            compression. Default 9.
    """
    mem = memoryview(image)
    width, height, format = get_im_props_from_memoryview(mem)
    with PngWriter(file, width, height, format, compression=compression) as writer:
        writer.write_static_frame(mem)


def write_apng(file, images, compression=9):
    """ "Write images to an apng file (animated png)."""

    frame_count = len(images)
    image_iter = iter(images)
    first_image = next(image_iter)

    width, height, format = get_im_props_from_memoryview(memoryview(first_image))

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
        chunk_limmit=2**20,
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
        if self.format not in FORMAT_TO_PSIZE or self.format == "p":
            raise ValueError(f"Unknown/unsupported color format: '{format}'")

        # Configuration
        self._compression = int(compression)
        self._png_mode = str(png_mode)
        self._frame_count = int(frame_count)  # for apng
        self._chunk_limit = int(chunk_limmit)

        # State
        self._animation_frames_written = 0
        self._apng_sequence_number = itertools.count()
        self._idat_written = 0  # 0 no, 1 wip, 2 done
        self._compressor = FrameCompressor(self._compression, self.height)

    def __enter__(self):
        # Write signature and header chunk
        self._write_signature()
        self._write_chunk_ihdr()
        if self._png_mode == "apng":
            self._write_chunk_actl()
        return self

    def __exit__(self, value, type, tb):
        self._write_chunk_iend()
        if self.fh is not self.file:
            self.fh.close()
        self.fh = None

        # Sanity checks
        if self._idat_written == 0:
            logger.warning("IDAT data has not been written.")
        elif self._idat_written == 1:
            logger.warning("Not all IDAT data has been written.")
        if self._png_mode == "apng":
            if self._animation_frames_written != self._frame_count:
                logger.warning(
                    f"PNGWriter has frame_count {self._frame_count} "
                    + "but wrote {self._animation_frames_written} animation frames."
                )

    def write_static_frame(self, image_data):
        """Write the reference image as a whole or in parts.

        Can be called multiple times with partial image data.
        """
        if self.fh is None:
            raise RuntimeError("Attempt to write to PngWriter after it has finished.")
        if self._idat_written == 2:
            raise RuntimeError("Static image is already written.")
        self._idat_written = 1

        mem = memoryview(image_data)
        w, _h, format = get_im_props_from_memoryview(mem)

        if format != self.format:
            raise RuntimeError("Image data format does not match the image format.")
        if w != self.width:
            raise RuntimeError("Image data width does not match the image width.")

        self._write_frame(self._compressor, mem, "IDAT")

    def _write_frame(self, compressor, mem, chunkname):
        compressor = self._compressor
        for i in range(mem.shape[0]):
            compressor.compress_scanline(mem[i : i + 1])
            if compressor.pending_nbytes >= self._chunk_limit:
                self._write_chunk_idat_or_fdat(chunkname, compressor.take_data())
        if compressor.done:
            self._write_chunk_idat_or_fdat(chunkname, compressor.take_data())
            self._idat_written = 2
        if compressor.too_much:
            logger.warning("Attempt to write image data beyond image height.")

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
        """Write a single animation frame.

        Must be called once for each frame.
        """
        if self._png_mode != "apng":
            raise RuntimeError("Not an apng file")
        if self.fh is None:
            raise RuntimeError("Attempt to write to PngWriter after it has finished.")
        if self._idat_written == 1:
            raise RuntimeError("Static image is in progress of being written.")

        mem = memoryview(image)
        w, h, format = get_im_props_from_memoryview(mem)

        if format != self.format:
            raise RuntimeError("Frame format does not match the image format.")

        x_offset, y_offset = int(x_offset), int(y_offset)
        if (
            x_offset < 0
            or y_offset < 0
            or x_offset + w > self.width
            or y_offset + h > self.height
        ):
            raise RuntimeError("Animation frame must be contained in reference image.")

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
            w, h, x_offset, y_offset, delay_num, delay_den, dispose_op, blend_op
        )

        # First frame can be the static or be independent
        if not self._idat_written:
            if w != self.width or h != self.height:
                raise RuntimeError("First animation frame must be full size.")
            self.write_static_frame(mem)
        else:
            compressor = FrameCompressor(self._compression, h)
            self._write_frame(compressor, mem, "fdAT")
            if not compressor.done:
                logger.warning("Not all fdAT data has been written.")

    def _write_signature(self):
        self.fh.write(b"\x89PNG\x0d\x0a\x1a\x0a")  # signature

    def _write_chunk(self, name: str, datas: Union[bytes, list]):
        if isinstance(datas, bytes):
            datas = [datas]
        bname = name.encode("ASCII")
        # Get checksum and byte count
        checksum = zlib.crc32(bname)
        nbytes = 0
        for bb in datas:
            nbytes += len(bb)
            checksum = zlib.crc32(bb, checksum)
        # Write
        self.fh.write(struct.pack(">I", nbytes))
        self.fh.write(bname)
        for bb in datas:
            self.fh.write(bb)
        self.fh.write(checksum.to_bytes(4, "big"))

    def _write_chunk_ihdr(self):
        # The image header
        bit_depth = 8
        color_type = FORMAT_TO_COLOR_FLAG[self.format]
        compression_method, filter_method, interlace_method = 0, 0, 0
        data = struct.pack(
            ">IIBBBBB",
            self.width,
            self.height,
            bit_depth,
            color_type,
            compression_method,
            filter_method,
            interlace_method,
        )
        self._write_chunk("IHDR", data)

    def _write_chunk_idat_or_fdat(self, name: str, datas: list):
        if name == "fdAT":
            datas.insert(0, struct.pack(">I", next(self._apng_sequence_number)))
        self._write_chunk(name, datas)

    def _write_chunk_iend(self):
        # The image trailer
        self._write_chunk("IEND", b"")

    def _write_chunk_actl(self):
        # The animation chunk control
        # Identifies that this is an animated png. The frame count must be exact.
        num_plays = 0  # loop indefinitely
        self._write_chunk("acTL", struct.pack(">II", self._frame_count, num_plays))

    def _write_chunk_fctl(self, *args):
        # The frame control chunk
        # width, height, x_offset, y_offset, delay_num, delay_den, dispose_op, blend_op
        data = struct.pack(">IIIIIHHBB", next(self._apng_sequence_number), *args)
        self._write_chunk("fcTL", data)


class FrameCompressor:
    """Convert scanlines to compressed data."""

    def __init__(self, compression, height):
        self._compressor = zlib.compressobj(compression, 8, 15, 9)
        self._height = height
        self._count = 0
        self.pending_nbytes = 0
        self.pending_data = []
        self.done = False
        self.too_much = False

    def compress_scanline(self, scanline):
        if self._count >= self._height:
            self.too_much = True
            return
        self._count += 1

        bb = self._compressor.compress(b"\x00")  # filter flag
        if bb:
            self.pending_data.append(bb)
            self.pending_nbytes += len(bb)
        bb = self._compressor.compress(scanline)
        if bb:
            self.pending_data.append(bb)
            self.pending_nbytes += len(bb)

        if self._count >= self._height:
            self.done = True
            bb = self._compressor.flush()
            if bb:
                self.pending_data.append(bb)
                self.pending_nbytes += len(bb)

    def take_data(self):
        data = self.pending_data
        self.pending_data = []
        self.pending_nbytes = 0
        return data
