#include "img_gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>

#include <fstream>

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

ImgSeqGen::ImgSeqGen():_width(0),_height(0),_deep(0),_min(-65535.0f),_max(65535.0f)
{

}

ImgSeqGen::~ImgSeqGen()
{

}

void ImgSeqGen::set_raw_data(const std::string& path , int width, int height , int deep)
{
    std::ifstream in(path.c_str() , std::ios::in | std::ios::binary);
    if(in.is_open()){
        _data.reset(new unsigned char[width*height*deep*2]);
        in.read((char*)_data.get() ,2*width*height*deep );
        _width = width;
        _height = height;
        _deep = deep;
        in.close();

        unsigned short *raw = (unsigned short*)_data.get();
        for(int i = 0 ;i<width*height ; ++i){
            unsigned short val = ntohs(raw[i]);
            if(val > _max){
                _max = static_cast<float>(val);
            }
            if(val < _min){
                _min = static_cast<float>(val);
            }
        }
    }
}

unsigned char* ImgSeqGen::gen_img(int slice)
{
    unsigned char* img_buffer = new unsigned char[_width*_height*4];
    memset(img_buffer , 9 , _width*_height*4);
    if(_data && slice > 0 && slice < _deep-1){
        unsigned short* img_data = (unsigned short*)_data.get() + slice*_width*_height;
        const float ww = 2352.0f;
        const float wl = 1179.0f;
        const float gray_min = wl - ww*0.5f; 
        for(int i =0 ; i<_width*_height ; ++i){
            float gray = (static_cast<float>(ntohs(img_data[i])) - gray_min)/ww * 255.0f;
            gray = gray > 255.0f ? 255.0f: gray;
            gray = gray < 0.0f ? 0.0f: gray;
            unsigned char ugray = static_cast<unsigned char>(gray);
            img_buffer[i*4] = ugray;
            img_buffer[i*4+1] = ugray;
            img_buffer[i*4+2] = ugray;
            img_buffer[i*4+3] = 255;
        }
        
    }

    return img_buffer;
}