#ifndef MED_IMAGING_IMAGE_DATA_H
#define MED_IMAGING_IMAGE_DATA_H

#include "MedImgIO/mi_io_stdafx.h"
#include "MedImgCommon/mi_common_define.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_matrix4.h"

MED_IMAGING_BEGIN_NAMESPACE

class IO_Export ImageData
{
public:
    ImageData();

    ~ImageData();

    bool mem_allocate();

    float get_min_scalar();

    float get_max_scalar();

    bool regulate_normalize_wl(float& fWindow, float& fLevel);

    bool regulate_wl(float& fWindow, float& fLevel);

    void normalize_wl(float& fWindow, float& fLevel);

    void get_pixel_value(unsigned int x, unsigned int y , unsigned int z , double& pValue) const;

    void get_pixel_value(const Point3& ptPos ,  double& pPixelValue) const;

    void set_data_dirty();

    void* get_pixel_pointer();

    void shallow_copy(ImageData *&pImgData);

    void deep_copy(ImageData *&pImgData);

public:
    DataType m_eDataType;
    unsigned int  m_uiChannelNum;
    unsigned int m_uiDim[3]; 
    double m_dSpacing[3]; 

    float m_fSlope;
    float m_fIntercept;
    Vector3 m_vImgOrientation[3];
    Point3 m_ptImgPositon;

private:
    float m_fMinScalar; 
    float m_fMaxScalar;
    bool   m_bCalcMinMax;
    std::unique_ptr<char[]> m_pMappedData;

private:
    template<typename T>
    void find_min_max_i(T *pData);

    void find_min_max_i();

    size_t get_data_size_i();
};

template<typename T>
void ImageData::find_min_max_i( T *pData )
{
    if(nullptr == pData)
    {
        return;
    }
    T min =  *pData, max = *pData;
    size_t size = m_uiDim[0] * m_uiDim[1] * m_uiDim[2] * m_uiChannelNum;

    for(int i = 1;i < size; i++)
    {
        T temp = *(pData + i);
        min = temp < min ? temp : min;
        max = temp > max ? temp : max;
    }

    m_fMinScalar = static_cast<float>(min);
    m_fMaxScalar = static_cast<float>(max);
}

MED_IMAGING_END_NAMESPACE

#endif