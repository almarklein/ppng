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

import io
import zlib
import struct
import pathlib
import logging
import itertools
from typing import Union

__all__ = ["PngReader", "PngWriter", "read_png", "write_apng", "write_png"]

__version__ = "1.0.0"
version_info = tuple(int(i) if i.isnumeric() else i for i in __version__.split("."))

logger = logging.getLogger("ppng")


# ----- utils


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

SINGLETON_CHUNKS = """ IHDR IEND PLTE acTL cHRM cICP gAMA iCCP mDCv cLLi sBIT sRGB
                       bKGD hIST tRNS eXIf pHYs tIME""".split()


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
    width, height, format = _get_im_props_from_memoryview(mem)
    with PngWriter(file, width, height, format, compression=compression) as writer:
        writer.write_static_frame(mem)


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
        # todo: check that all is written
        self._write_chunk_iend()
        if self.fh is not self.file:
            self.fh.close()
        self.fh = None

        # Sanity checks
        if not self._compressor.done:
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
        w, _h, format = _get_im_props_from_memoryview(mem)

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
        w, h, format = _get_im_props_from_memoryview(mem)

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
    """Go from scanlines to compressed data."""

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


# ----- reading


def read_png(file):
    """Read a png file.

    Parameters:
        file : str | pathlib.Path | file-like
            The file to read from. Can be a filename or anything that has a
            ``read()`` method.

    Returns:
        image : memoryview
            The image as a multidimensional array.
            Can be converted to Numpy using ``np.asarray(image)``.
            The shape is ``(h, w)`` for grayscale, ``(h, w, 2)`` for grayscale+alpha,
            ``(h, w, 3)`` for RGB, and ``(h, w, 4)`` for RGBA.
    """

    reader = PngReader(file)

    # Get metadata
    info = reader.read_meta()
    w, h = info["width"], info["height"]
    psize = FORMAT_TO_PSIZE[info["format"]]
    nbytes = w * h * psize * info["bit_depth"] // 8
    row_stride = w * psize

    # Allocate image
    shape = (h, w, psize) if psize > 1 else (h, w)
    image = memoryview(bytearray(nbytes)).cast("B")  # 1D

    # Read data
    y = 0
    for scanlines in reader.iter_image_chunks():
        for scanline in scanlines:
            image[y * row_stride : (y + 1) * row_stride] = scanline
            y += 1

    return image.cast("B", shape)


class PngReader:
    def __init__(self, file):
        # Get file handle
        self.file = file
        if isinstance(self.file, (str, pathlib.Path)):
            self.fh = open(self.file, "rb")
        elif isinstance(file, bytes):
            self.fh = io.BytesIO(file)
        elif hasattr(self.file, "read"):
            self.fh = self.file
        else:
            raise RuntimeError(f"Cannot read from {self.file}")

        # Variables
        self._next_chunk = None  # for peeking
        self._header_info = None  # needed to process further chunks
        self._chunk_counts = {}  # keep track of chunk appearances
        self._decompressor = None  # the current image decompressor

        # Start
        self._read_signature()

    # todo: somewhere check that no interlacing

    def read_meta(self):
        """Get a dict of metadata.

        Walks over the chunks, collecting information from IHDR and other chunks,
        until a chunk is reached that signals the start of the data.
        """

        info = {}
        while True:
            name, _ = self._peek_chunk()
            if name in ("IDAT", "fcTL", "fdAT", "IEND"):
                break
            chunk = self.read_chunk()
            for error in chunk["errors"]:
                logger.warning("PNG read error: " + error)
            if "info" in chunk:
                info.update(chunk["info"])
        return info

    def iter_image_chunks(self):
        """Generator to read the static frame (i.e. default image).

        Yields lists of scanlines (bytearray objects) that must be concatenated
        by the caller to create the final image.
        """

        # Move to the first IDAT chunk
        while True:
            name, _ = self._peek_chunk()
            if name == "IDAT":
                break
            if name == "IEND":
                raise RuntimeError("IDAT chunk not found")
            self.read_chunk()  # skip

        # Read subsequent IDAT chunks
        while True:
            name, _ = self._peek_chunk()
            if name != "IDAT":
                break
            chunk = self.read_chunk()
            for error in chunk["errors"]:
                logger.warning("PNG read error: " + error)
            yield chunk["scanlines"]

    def iter_animation_images(self):
        """Generator to read animation frames.

        Yields each animation image.
        Images that consist of multiple chunks are concatenated for you.
        """

        # todo: make sure that this is indeed apng
        # todo: ancillary chunks can be in arbitrary order :/

        while True:
            name, _ = self._peek_chunk()
            if name == "fcTL":
                pass
            if name == "IEND":
                raise RuntimeError("IDAT chunk not found")
            self.read_chunk()  # skip

    def read_chunk(self):
        """Read a chunk from the PNG file.

        This method does not fail when the chunk is wrong, so it can be used to
        examine a broken png file (unless the file is corrupt in a way that
        chunks cannot be detected).

        Returns a dict with metadata. For IDAT and fdAT chunks it will also
        contain "scanlines". The dict also has a key "errors" that may contain
        strings indicating problems with the chunk. If it contains the key
        "ignore", then ...
        """
        name, data = self._read_chunk()
        if name == "EOF":
            raise RuntimeError("Attempt to read past the end of the file")
        next_name, _ = self._peek_chunk()

        errors = []
        result = {"name": name, "errors": errors}

        self._chunk_counts[name] = self._chunk_counts.get(name, 0) + 1

        # Many chunks are only allowed once

        if name in SINGLETON_CHUNKS:
            if self._chunk_counts[name] > 1:
                errors.append(f"{name} chunk is present more than once.")
                result["ignore"] = True
        # All chunks must be between IHDR and IEND
        if name not in ("IHDR", "IEND"):
            if "IHDR" not in self._chunk_counts:
                errors.append("Chunk before IHDR chunk.")
            if "IEND" in self._chunk_counts:
                errors.append("Chunks after IEND chunk.")
        # Final check
        if name in ("IEND", "EOF"):
            for n in ("IHDR", "IEND", "IDAT"):
                if n not in self._chunk_counts:
                    errors.append(f"Chunk {n} not present.")

        try:
            if result.get("ignore", False):
                pass
            elif name == "IHDR":
                w, h, d, ct, cm, fm, im = struct.unpack(">IIBBBBB", data)
                format = COLOR_FLAG_TO_FORMAT.get(ct, None)
                if cm != 0:
                    errors.append("Expected compression_method to be 0.")
                if fm != 0:
                    errors.append("Expected filter_method to be 0.")
                self._header_info = {
                    "width": w,
                    "height": h,
                    "bit_depth": d,
                    "format": format,
                    "interlace_method": im,
                }
                result["info"] = self._header_info
            elif name == "IEND":
                pass
            elif name == "IDAT":
                if self._decompressor is None:
                    self._decompressor = FrameDecompressor(self._header_info)
                    result["scanlines"] = self._decompressor.decompress(data)
                if next_name != "IDAT":
                    result["scanlines"] += self._decompressor.flush()
                    if not self._decompressor.done:
                        errors.apppend("There was not enough data in the IDAT chunks.")
                    self._decompressor = None
            elif name == "acTL":
                pass
            elif name == "fcTL":
                pass
            elif name == "fdAT":
                pass
        except Exception as err:
            errors.append(str(err))

        return result

    def _read_signature(self):
        sig_ref = b"\x89PNG\x0d\x0a\x1a\x0a"
        sig = self.fh.read(len(sig_ref))
        if sig != sig_ref:
            raise RuntimeError("PNG signature invalid")

    def _peek_chunk(self):
        if self._next_chunk is None:
            bb = self.fh.read(4)
            if not bb:
                self._next_chunk = "EOF", 0
            else:
                (nbytes,) = struct.unpack(">I", bb)
                name = self.fh.read(4).decode()
                self._next_chunk = name, nbytes
        return self._next_chunk

    def _read_chunk(self):
        name, nbytes = self._peek_chunk()
        self._next_chunk = None

        data = self.fh.read(nbytes)
        ref_checksum = self.fh.read(4)

        checksum = zlib.crc32(data, zlib.crc32(name.encode("ASCII")))
        checksum = checksum.to_bytes(4, "big")
        if checksum != ref_checksum:
            raise RuntimeError(f"Checksum in chunk {name} does not match.")

        return name, data


class FrameDecompressor:
    """To go from compressed bytes to scanlines."""

    def __init__(self, header_info):
        self._info = header_info
        self._decompressor = zlib.decompressobj(15)
        self._pending_bytes = None
        self._last_scanline = None
        self._scanline_count = 0
        self._height = self._info["height"]
        self.done = False

    def decompress(self, data):
        decompressed = self._decompressor.decompress(data)
        # If a writer separately compressed its chunks, don't fail
        if self._decompressor.eof:
            tail = self._decompressor.unused_data
            self._decompressor = zlib.decompressobj(15)
            decompressed += self._decompressor.decompress(tail)
        return self._process(decompressed)

    def flush(self):
        decompressed = self._decompressor.flush()
        return self._process(decompressed)

    def _process(self, decompressed):
        if self._pending_bytes:
            decompressed = self._pending_bytes + decompressed
            self._pending_bytes = None
        decompressed = memoryview(decompressed)
        # Convert data to scanlines
        psize = FORMAT_TO_PSIZE[self._info["format"]]
        stride = self._info["width"] * psize
        stride = stride * self._info["bit_depth"] // 8 + 1
        scanlines = []
        while len(decompressed) >= stride and self._scanline_count < self._height:
            raw_scanline, decompressed = decompressed[:stride], decompressed[stride:]
            scanline = unfilter_scanline(raw_scanline, psize, self._last_scanline)
            scanlines.append(scanline)
            self._last_scanline = scanline
            self._scanline_count += 1
        if decompressed.nbytes > 0:
            # Chunk boundaries can be mid-scanline
            self._pending_bytes = decompressed.tobytes()
        if self._scanline_count >= self._height:
            self.done = True
        return scanlines


def unfilter_scanline(line_bytes, fu=4, prev=None):
    """Scanline unfiltering, inspired by pypng"""
    # From what I found, the filtering is relatively costly while barely adding
    # a compression benefit. This is why we don't use it in the writer.

    # todo: this code can be optimized some more

    filter = line_bytes[0]
    line1 = memoryview(line_bytes)[1:]  # avoid making a copy
    # todo: cast to 16 bit if bithdepth is so
    if filter == 0:
        return line1  # no filter
    line2 = bytearray(line1)  # copy for output

    if filter == 1:
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
