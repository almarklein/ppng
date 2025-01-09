# PNG test suite

The "official" test-suite for PNG, from http://www.schaik.com/pngsuite.

## File naming

Where possible, the test-files are 32x32 bits icons. This results in a still reasonable size of the suite even with a large number of tests. The name of each test-file reflects the type in the following way:

    filename:                               g04i2c08.png
                                            || ||||
    test feature (in this case gamma) ------+| ||||
    parameter of test (here gamma-value) ----+ ||||
    interlaced or non-interlaced --------------+|||
    color-type (numerical) ---------------------+||
    color-type (descriptive) --------------------+|
    bit-depth ------------------------------------+


color-type:

- 0g - grayscale
- 2c - rgb color
- 3p - paletted
- 4a - grayscale + alpha channel
- 6a - rgb color + alpha channel

bit-depth:

- 01 - with color-type 0, 3
- 02 - with color-type 0, 3
- 04 - with color-type 0, 3
- 08 - with color-type 0, 2, 3, 4, 6
- 16 - with color-type 0, 2, 4, 6

interlacing:

- n - non-interlaced
- i - interlaced


## License

Permission to use, copy, modify and distribute these images for any
purpose and without fee is hereby granted.

(c) Willem van Schaik, 1996, 2011
