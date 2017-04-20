#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_sampler.h"
#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"
#include <limits>

using namespace MED_IMAGING_NAMESPACE;

void main()
{
    Vector3 vvv(12,3.4f,5);

    double d = 2.56;
    double d2 = ArithmeticUtils::FloorDouble(d);

    size_t iii = sizeof(RGBAUnit);

    Vector3f v0(1,3,7);
    Vector3f v1(2,1,5);
    Vector3f v3 = min_per_elem(v0,v1);
    Vector3f v4 = max_per_elem(v0,v1);

    Vector3f v5 = v0.max_per_elem(v1);

    Vector3f v6(-0.0, 1.6595 , -std::numeric_limits<float>::max());
    Vector3f v7 = v6.to_abs();
    Vector3f v8 = v7.less_than(Vector3f(10.0f , FLOAT_EPSILON , FLOAT_EPSILON));
    float f = std::numeric_limits<float>::max()*0.5f + std::numeric_limits<float>::max()*0.5f;

    //Test 1D sampler
    unsigned short* pData = new unsigned short[10];
    for (int i = 0 ; i <10 ; ++i)
    {
        pData[i] = unsigned short(i);
    }

    Sampler<unsigned short> sample;
    std::cout  << sample.sample_1d_nearst(3.4f , 10 , pData) << std::endl;
    std::cout  << sample.sample_1d_linear(3.4f , 10 , pData) << std::endl;

    //Test 2D sampler
    //std::ifstream in("D:/AB_CTA_01_0.raw", std::ios::binary | std::ios::out);
    //unsigned short* pImg = new unsigned short[512*512];
    //in.read((char*)pImg , 512*512*2);
    //in.close();

    //unsigned int uiWidth  = 1024;
    //unsigned int uiHeight = 1024;
    //unsigned short *pImgZoom = new unsigned short[uiWidth*uiHeight];
    //for (unsigned int y =0 ; y < uiHeight ;++y)
    //{
    //    for (unsigned int x = 0 ; x < uiWidth ; ++x)
    //    {
    //        if (x == 18 && y == 1022)
    //        {
    //            std::cout << "EE";
    //        }
    //        float fX = (float)x/(float)(uiWidth) * 512;
    //        float fY = (float)y/(float)(uiHeight) * 512;
    //        pImgZoom[y*uiWidth + x] = unsigned short(sample.sample_2d_linear(fX ,fY, 512,512 , pImg));
    //    }
    //}

    //std::ofstream out("D:/AB_CTA_01_0_zoom.raw", std::ios::binary | std::ios::out);
    //out.write((char*)pImgZoom,1024*1024*2);
    //out.close();

    //unsigned short usMax = 0;
    //unsigned int uMaxX = 0 ,  uMaxY = 0;
    //for (unsigned int y =0 ; y < uiHeight ;++y)
    //{
    //    for (unsigned int x = 0 ; x < uiWidth ; ++x)
    //    {
    //        if (pImgZoom[y*uiWidth + x] > usMax)
    //        {
    //            usMax = pImgZoom[y*uiWidth + x];
    //            uMaxX = x;
    //            uMaxY = y;
    //        }
    //    }
    //}

    //delete [] pImg;
    //delete [] pImgZoom;

    //Test 3D sampler
    unsigned int uiX = 256;
    unsigned int uiY = 256;
    unsigned int uiZ = 256;
    std::ifstream in("D:/AB_CTA_01_256.raw", std::ios::binary | std::ios::out);
    unsigned short* pImg = new unsigned short[uiX*uiY*uiZ];
    in.read((char*)pImg , uiX*uiY*uiZ*2);
    in.close();

    unsigned int uiWidth  = 800;
    unsigned int uiHeight = 800;
    unsigned int uiDepth = 800;
    unsigned short *pImgZoom = new unsigned short[uiWidth*uiHeight*uiDepth];
    for (unsigned int z = 0 ; z < uiDepth ; ++z)
    {
        for (unsigned int y =0 ; y < uiHeight ;++y)
        {
            for (unsigned int x = 0 ; x < uiWidth ; ++x)
            {
                float fX = (float)x/(float)(uiWidth) * uiX;
                float fY = (float)y/(float)(uiHeight) * uiY;
                float fZ = (float)z/(float)(uiDepth) * uiZ;
                pImgZoom[z*uiWidth*uiHeight + y*uiWidth + x] = (unsigned short)(sample.sample_3d_nearst(fX ,fY, fZ , uiX,uiY , uiZ ,pImg));
            }
        }
    }
    

    std::ofstream out("D:/AB_CTA_01_zoom.raw", std::ios::binary | std::ios::out);
    out.write((char*)pImgZoom,uiWidth*uiWidth*uiDepth*2);
    out.close();

    std::cout << "Done\n";


}