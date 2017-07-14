#ifndef TEST_IMA_GEN_H
#define TEST_IMA_GEN_H

#include <memory>
#include <string>

class ImgGen
{
public:
    ImgGen();
    ~ImgGen();

    unsigned char* gen_img(int width , int height);
};

class ImgSeqGen
{
public:
    ImgSeqGen();
    ~ImgSeqGen();

    void set_raw_data(const std::string& path , int width, int height , int deep);
    
    unsigned char* gen_img(int slice);

private:
    std::unique_ptr<unsigned char[]> _data;
    int _width;
    int _height;
    int _deep;
    float _min;
    float _max;
};

#endif
