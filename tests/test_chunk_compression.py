import zlib


def decompress_simple(*chunks):
    d = zlib.decompressobj(15)
    decompressed = b""

    for chunk in chunks:
        decompressed += d.decompress(chunk)

    decompressed += d.flush()
    return decompressed


def decompress_advanced(*chunks):
    d = zlib.decompressobj(15)
    decompressed = b""

    for chunk in chunks:
        decompressed += d.decompress(chunk)
        if d.eof:
            tail = d.unused_data
            d = zlib.decompressobj(15)
            decompressed += d.decompress(tail)

    decompressed += d.flush()
    return decompressed


def test_compression_boundaries1():
    # A single chunk

    c = zlib.compressobj(9, 8, 15, 9)

    chunk1 = c.compress(b"123")
    chunk1 += c.compress(b"456")
    chunk1 += c.flush()

    result = decompress_simple(chunk1)
    assert result == b"123456"

    result = decompress_advanced(chunk1)
    assert result == b"123456"


def test_compression_boundaries2():
    # Two chunks, as intended

    c = zlib.compressobj(9, 8, 15, 9)

    chunk1 = c.compress(b"123")

    chunk2 = c.compress(b"456")
    chunk2 += c.flush()

    result = decompress_simple(chunk1, chunk2)
    assert result == b"123456"

    result = decompress_advanced(chunk1, chunk2)
    assert result == b"123456"


def test_compression_boundaries3():
    # Two chunks, separate compressions

    c = zlib.compressobj(9, 8, 15, 9)

    chunk1 = c.compress(b"123")
    chunk1 += c.flush()

    c = zlib.compressobj(9, 8, 15, 9)

    chunk2 = c.compress(b"456")
    chunk2 += c.flush()

    result = decompress_simple(chunk1, chunk2)
    assert result == b"123"  # incomplete!

    result = decompress_advanced(chunk1, chunk2)
    assert result == b"123456"
