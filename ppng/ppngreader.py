# Copyright (c) 2016-2025, Almar Klein
# This module is distributed under the terms of the MIT License.
"""
Pure python module for reading png images.
"""

__all__ = ["PngReader", "read_png"]

import io
import zlib
import struct
import pathlib

from .ppngwriter import logger, FORMAT_TO_PSIZE, COLOR_FLAG_TO_FORMAT


SINGLETON_CHUNKS = """ IHDR IEND PLTE acTL cHRM cICP gAMA iCCP mDCv cLLi sBIT sRGB
                       bKGD hIST tRNS eXIf pHYs tIME""".split()


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
    """Convert compressed data to scanlines."""

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

    if filter in (2, 3, 4) and prev is None:
        prev = bytearray(len(line1))  # zeros

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
