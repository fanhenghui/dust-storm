#pragma once

#include "Ext/gl/glew.h"
#include <string>

/*
OFFSET              Count TYPE   Description
0000h                   1 byte   Length of image identification field (below)
0001h                   1 byte   Color map type :
0 - no color map
1 - 256 entry palette
0002h                   1 byte   Image type :
0 - no image data included
1 - Uncompressed, color-mapped image
2 - Uncompressed, RGB image
3 - Uncompressed, black and white image
9 - RLE encoded color-mapped image
10 - RLE encoded RGB image
11 - Compressed, black and white image
32 - Compressed color-mapped data, using
Huffman, Delta, and runlength encoding.
33 - Compressed color-mapped data, using
Huffman, Delta, and RLE.  4-pass quadtree-
type process.
0003h                   1 word   Index of first color map entry
0005h                   1 word   Count of color map entries
0007h                   1 byte   Number of bits per color map entry
0008h                   1 word   X coordinate of the lower left corner of
the image.
000Ah                   1 word   Y coordinate of the lower left corner of
the image.
000Ch                   1 word   Width of the image in pixels
000Eh                   1 word   Height of the image in pixels
0010h                   1 byte   Bytes per pixel
0011h                   1 byte   Flags (bitmapped):
0-3 : Number of attribute bits
4 : reserved
5 : Screen origin in upper left corner
6-7 : Data storage interleave
00 - no interleave
01 - even/odd interleave
10 - four way interleave
11 - reserved
The byte should be set to 0. Don't know why.
0012h                   ? char   Image identification string, usually not there,
when the length (see up) is 0.
????h                   ? byte   Color map data
Depending on the number of bits per color map
entry, the entries here have a different size.
4 bytes : 1 byte for blue
1 byte for green
1 byte for red
1 byte for attribute
3 bytes : 1 byte for blue
1 byte for green
1 byte for red
2 bytes : Bitmapped as a word in Intel byte
order as follows :
ARRRRRGG GGGBBBBB
????h                   ? byte   Image data
For images of type 9 (using RLE), the image
data is divided into packets, the first byte
being the indicator for repetition or copy.
If bit 7 of the first byte is set, then repeat
(first byte and 07Fh+1) times the next byte,
otherwise copy first byte+1 pixels from data
stream. RLE packets may cross scan lines !
*/

typedef struct
{
    unsigned char identsize;              // Size of ID field that follows header (0)
    unsigned char colorMapType;           // 0 = None, 1 = palette
    unsigned char imageType;              // 0 = none, 1 = indexed, 2 = rgb, 3 = grey, +8=rle
    unsigned short colorMapStart;          // First color map entry
    unsigned short colorMapLength;         // Number of colors
    unsigned char  colorMapBits;   // bits per palette entry
    unsigned short xstart;                 // image x origin
    unsigned short ystart;                 // image y origin
    unsigned short width;                  // width in pixels
    unsigned short height;                 // height in pixels
    unsigned char bits;                   // bits per pixel (8 16, 24, 32)
    unsigned char descriptor;             // image descriptor
} TGAHEADER;

class Targa
{
public:
     static GLbyte* ReadTGABits(const char *szFileName, GLint& iWidth, GLint& iHeight, GLint& iComponents, GLenum& eFormat);
     static void WriteRGBImageToTGA(const char* szFileName, const GLbyte* pBits, GLint iWidth, GLint iHeight);
protected:
private:
};
