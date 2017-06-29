#include "img_gen.h"

ImgGen::ImgGen()
{

}

ImgGen::~ImgGen()
{

}

unsigned char* ImgGen::gen_img(int width , int height )
{
    static int tag = 0;
    tag += 5;
    if(tag > 255) {
        tag = 0;
    }

    unsigned char* buffer = new unsigned char[width*height*4];
    for(int i = 0 ;i< width*height ; ++i){
        buffer[i*4] = tag;
        buffer[i*4+1] = tag;
        buffer[i*4+2] = tag;
        buffer[i*4+3] = 255;
    }

    return buffer;
    


}