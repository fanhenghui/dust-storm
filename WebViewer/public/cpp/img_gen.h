#ifndef TEST_IMA_GEN_H
#define TEST_IMA_GEN_H


class ImgGen
{
public:
    ImgGen();
    ~ImgGen();

    unsigned char* gen_img(int width , int height);
};


#endif
