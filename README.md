# ppng

Lightweight png reader and writer

The first P in `ppng` stands for a number of things:

* Pure: the code is pure Python and has zero dependencies.
* Pretty: a moderately sized single module.
* Performant: the heavy lifting is done with Python's builtin zlib module.
  Further it avoids memory copies.
  And the writer does not do fancy filtering that is slow and barely reduces the file size.
* Powerful: supports apng, streamed reading and writing, etc.
* Partial: does not support the full png spec (see below).


## PNG is pretty cool

PNG is a widespread format to store lossless images. It offers good compression, pretty nice
features with e.g. 16 bit images, is extensible, all with a relatively simple spec (simple enough to implement in pure Python).
The animated version (APNG) is also widely supported nowadays which has many advantages over GIF.

New formats like WEWBP and AVIF offer better compression, but these are much more complex,
so you basically rely on the library implementation of e.g. Google. Also, these formats are
less flexible, e.g. no streamed writing of massive images.


## Support

Supported:
* grayscale, grayscale + alpha, rgb , rgba.
* 8 or 16 bit (TODO)
* APNG
* Streamed reading / writing

The writer is (deliberately) kept simple. It does not support:
* Paletted images.
* Interlacing.
* Pre-compression filtering.

The reader can read anything that ppng can write, and more. But it does not support:
* Paletted images.
* Interlaced images.

Maybe the restrictions in the reader can be released?

Extensions to PNG:
* Volumetric images (TODO)
* Multichannel images (TODO)
