#include "targa.h"
#include <fstream>
#include <exception>

////////////////////////////////////////////////////////////////////
// Allocate memory and load targa bits. Returns pointer to new buffer,
// height, and width of texture, and the OpenGL format of data.
// Call free() on buffer when finished!
// This only works on pretty vanilla targas... 8, 24, or 32 bit color
// only, no palettes, no RLE encoding.
GLbyte *Targa::ReadTGABits(const char *szFileName, GLint& iWidth, GLint& iHeight, GLint& iComponents, GLenum& eFormat)
{
    FILE *pFile;			// File pointer
    TGAHEADER tgaHeader;		// TGA file header
    unsigned long lImageSize;		// Size in bytes of image
    short sDepth;			// Pixel depth;
    GLbyte	*pBits = NULL;          // Pointer to bits

                                    // Default/Failed values
    iWidth = 0;
    iHeight = 0;
    eFormat = GL_RGB;
    iComponents = GL_RGB;

    // Attempt to open the file
    fopen_s(&pFile, szFileName, "rb");
    if (pFile == NULL)
        return NULL;
    // Read in header (binary)
    unsigned char aHead[18];
    fread(aHead, 18, 1, pFile);
    tgaHeader.identsize = aHead[0];
    tgaHeader.colorMapType = aHead[1];
    tgaHeader.imageType = aHead[2];
    tgaHeader.colorMapStart = ((unsigned short(0) & aHead[4]) << 8 | (aHead[3]));
    tgaHeader.colorMapLength = ((unsigned short(0) & aHead[6]) << 8 | (aHead[5]));
    tgaHeader.colorMapBits = aHead[7];
    tgaHeader.xstart = ((unsigned short(0) | aHead[9]) << 8 | (aHead[8]));
    tgaHeader.ystart = ((unsigned short(0) | aHead[11]) << 8 | (aHead[10]));
    tgaHeader.width = ((unsigned short(0) | aHead[13]) << 8 | (aHead[12]));
    tgaHeader.height = ((unsigned short(0) | aHead[15]) << 8 | (aHead[14]));
    tgaHeader.bits = aHead[16];
    tgaHeader.descriptor = aHead[17];
    //fread(&tgaHeader, 18, 1, pFile);
    // Get width, height, and depth of texture
    iWidth = tgaHeader.width;
    iHeight = tgaHeader.height;
    sDepth = tgaHeader.bits / 8;

    // Put some validity checks here. Very simply, I only understand
    // or care about 8, 24, or 32 bit targa's.
    if (tgaHeader.bits != 8 && tgaHeader.bits != 24 && tgaHeader.bits != 32)
        return NULL;

    // Calculate size of image buffer
    lImageSize = tgaHeader.width * tgaHeader.height * sDepth;

    // Allocate memory and check for success
    pBits = (GLbyte*)malloc(lImageSize * sizeof(GLbyte));
    if (pBits == NULL)
        return NULL;

    // Read in the bits
    // Check for read error. This should catch RLE or other 
    // weird formats that I don't want to recognize
    if (fread(pBits, lImageSize, 1, pFile) != 1)
    {
        free(pBits);
        return NULL;
    }

    // Set OpenGL format expected
    switch (sDepth)
    {
#ifndef OPENGL_ES
    case 3:     // Most likely case
        eFormat = GL_BGR;
        iComponents = GL_RGB;
        break;
#endif
    case 4:
        eFormat = GL_BGRA;
        iComponents = GL_RGBA;
        break;
    case 1:
        eFormat = GL_LUMINANCE;
        iComponents = GL_LUMINANCE;
        break;
    default:        // RGB
                    // If on the iPhone, TGA's are BGR, and the iPhone does not 
                    // support BGR without alpha, but it does support RGB,
                    // so a simple swizzle of the red and blue bytes will suffice.
                    // For faster iPhone loads however, save your TGA's with an Alpha!
#ifdef OPENGL_ES
        for (int i = 0; i < lImageSize; i += 3)
        {
            GLbyte temp = pBits[i];
            pBits[i] = pBits[i + 2];
            pBits[i + 2] = temp;
        }
#endif
        break;
    }

    // Done with File
    fclose(pFile);

    // Return pointer to image data
    return pBits;
}

void Targa::WriteRGBImageToTGA(
    const char* szFileName, const GLbyte* pBits, GLint iWidth, GLint iHeight)
{
    try
    {
        //目前只能存储非压缩的Tga RGB格式 
        // pBits 是RGB排列的裸数据
        //CHECK_PTR(pBits);

        //Create file header
        unsigned char aHead[18];
        memset(aHead, 18, 0);

        aHead[0] = 0;//
        aHead[1] = 0;//no color map
        aHead[2] = 2;//uncompressed rgb image

                     //no color map all zero
        aHead[3] = 0;
        aHead[4] = 0;
        aHead[5] = 0;
        aHead[6] = 0;
        aHead[7] = 0;

        aHead[8] = 0;
        aHead[9] = 0;

        aHead[10] = 0;
        aHead[11] = 0;

        unsigned short usWidth = static_cast<unsigned short>(iWidth);
        unsigned short usHeight = static_cast<unsigned short>(iHeight);

        aHead[12] = (unsigned char)usWidth;
        aHead[13] = (unsigned char)(usWidth >> 8);

        aHead[14] = (unsigned char)usHeight;
        aHead[15] = (unsigned char)(usHeight >> 8);

        aHead[16] = 24;
        aHead[17] = 0;

        //Wirte to file
        std::ofstream out;
        out.open(szFileName, std::ios::out | std::ios::binary);
        if (!out.is_open())
        {
            throw std::exception((std::string("Open file") + std::string(szFileName) + std::string("failed!")).c_str());
        }
        //Write info head
        out.write((char*)aHead, 18);
        out.write((char*)pBits, iWidth*iHeight * 3);

        out.close();
    }
    catch (const std::exception& e)
    {
        //LERROR(std::string("write RGB image to tga file failed! ") + e.what());
        throw e;
    }

}

