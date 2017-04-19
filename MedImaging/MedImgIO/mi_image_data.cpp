#include "mi_image_data.h"
#include "MedImgArithmetic/mi_sampler.h"

MED_IMAGING_BEGIN_NAMESPACE

ImageData::ImageData(): 
m_eDataType(SHORT)
    , m_uiChannelNum(1)
    , m_fMinScalar(0)
    , m_fMaxScalar(1024)
    , m_fSlope(1.0)
    , m_fIntercept(0.0)
    , m_bCalcMinMax(false)
{
    m_uiDim[0] = 0;
    m_uiDim[1] = 0;
    m_uiDim[2] = 0;

    m_dSpacing[0] = 1;
    m_dSpacing[1] = 1;
    m_dSpacing[2] = 1;

    m_vImgOrientation[0] = Vector3(1.0,0.0,0.0);
    m_vImgOrientation[1] = Vector3(0.0,1.0,0.0);
    m_vImgOrientation[2] = Vector3(0.0,0.0,1.0);

    m_ptImgPositon = Point3(0.0, 0.0, 0.0);
}

ImageData::~ImageData()
{

}

bool ImageData::AllocateMemory()
{
    const size_t imemSize = GetVolumeDataSize_i();
    m_pMappedData.reset(new char[imemSize]);
    m_bCalcMinMax = false;
    return true;
}

float ImageData::GetMinScalar()
{
    if(!m_bCalcMinMax)
    {
        CalculateMinMax_i();
    }
    return m_fMinScalar;
}

float ImageData::GetMaxScalar()
{
    if(!m_bCalcMinMax)
    {
        CalculateMinMax_i();
    }
    return m_fMaxScalar;
}

bool ImageData::RegulateWindowLevel(float& fWindow, float& fLevel)
{
    // CT should apply slope and intercept
    // MR has always slope(1) and intercept(0)
    if (m_fSlope < DOUBLE_EPSILON)
    {
        return false;
    }

    double dMin = GetMinScalar();

    fLevel = (fLevel - m_fIntercept)/m_fSlope;
    fWindow = fWindow/m_fSlope;

    return true;
}


void ImageData::NormalizeWindowLevel(float& fWindow, float& fLevel)
{
    const static float f65535R = 1.0f/65535.0f;
    const static float f255R = 1.0f/255.0f;

    float fMin = GetMinScalar();

    switch(m_eDataType)
    {
    case  USHORT:
        fWindow *= f65535R;
        fLevel *= f65535R;
        break;
    case  SHORT:
        fWindow *= f65535R;
        fLevel = (fLevel - std::min(0.0f, fMin) ) * f65535R;
        break;
    case UCHAR:
        fWindow *= f255R;
        fLevel *= f255R;
        break;
    case CHAR:
        fWindow *= f255R;
        fLevel = (fLevel - std::min(0.0f, fMin)) *f255R;
        break;
    case FLOAT:
        break;
    default:
        IO_THROW_EXCEPTION("Undefined image type!");
    }
}


bool ImageData::RegulateWindowLevelAndNormalize(float& fWindow, float& fLevel)
{
    if (!RegulateWindowLevel(fWindow , fLevel))
    {
        return false;
    }

    NormalizeWindowLevel(fWindow , fLevel);

    return true;
}

void ImageData::GetPixelValue(unsigned int x,unsigned int y ,unsigned int z , double& pValue) const
{
    void* pMappedData = m_pMappedData.get();
    if (nullptr == pMappedData)
    {
        IO_THROW_EXCEPTION("Undefined image type!");("Volume data is null!");
    }

    // check if the x,y,z coordinate is valid
    if ( x >= m_uiDim[0] ||
        y >= m_uiDim[1] ||
        z >= m_uiDim[2] )
    {
        IO_THROW_EXCEPTION("Input coordinate is out of data range!");
    }

    // Find the pixel value
    double dPixelValue = 0.0;
    unsigned int nOffset = x + y * m_uiDim[0] + z * m_uiDim[0] * m_uiDim[1];
    switch(m_eDataType)
    {
    case CHAR:
        {
            char *pChar = (char *)pMappedData;
            dPixelValue =(double) pChar[nOffset];
            break;
        }
    case UCHAR:
        {
            unsigned char *pUchar = (unsigned char *)pMappedData;
            dPixelValue =(double) pUchar[nOffset];
            break;
        }
    case USHORT:
        {
            unsigned short *pUshort = (unsigned short *)pMappedData;
            dPixelValue =(double) pUshort[nOffset];
            break;
        }
    case SHORT:
        {
            short *pShort = (short *)pMappedData;
            dPixelValue =(double) pShort[nOffset];
            break;
        }
    case FLOAT:
        {
            float *pFloat = (float *)pMappedData;
            dPixelValue =(double) pFloat[nOffset];
            break;
        }
    default:
        IO_THROW_EXCEPTION("Undefined image type!");
    }

    pValue = dPixelValue;
}

void ImageData::GetPixelValue(const Point3& ptPos , double& pValue) const
{
    void* pMappedData = m_pMappedData.get();
    if (nullptr == pMappedData)
    {
        IO_THROW_EXCEPTION("Undefined image type!");("Volume data is null!");
    }

    // check if the x,y,z coordinate is valid
    if ( ptPos.x > m_uiDim[0]-1 ||
        ptPos.y > m_uiDim[1]-1 ||
        ptPos.z > m_uiDim[2]-1 )
    {
        IO_THROW_EXCEPTION("Input coordinate is out of data range!");
    }

    // Find the pixel value
    double dPixelValue = 0.0;
    switch(m_eDataType)
    {
    case CHAR:
        {
            Sampler<char> sampler;
            dPixelValue =(double) sampler.Sample3DLinear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (char*)pMappedData);
            break;
        }
    case UCHAR:
        {
            Sampler<unsigned char> sampler;
            dPixelValue =(double) sampler.Sample3DLinear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (unsigned char*)pMappedData);
            break;
        }
    case USHORT:
        {
            Sampler<unsigned short> sampler;
            dPixelValue =(double) sampler.Sample3DLinear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (unsigned short*)pMappedData);
            break;
        }
    case SHORT:
        {
            Sampler<short> sampler;
            dPixelValue =(double) sampler.Sample3DLinear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (short*)pMappedData);
            break;
        }
    case FLOAT:
        {
            Sampler<float> sampler;
            dPixelValue =(float) sampler.Sample3DLinear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (float*)pMappedData);
            break;
        }
    default:
        IO_THROW_EXCEPTION("Undefined image type!");
    }
    pValue = dPixelValue;
}

void ImageData::SetImageDataDirty()
{
    m_bCalcMinMax = false;
}

void* ImageData::GetPixelPointer()
{
    return m_pMappedData.get();
}

void ImageData::ShallowCopy(ImageData *&pImgData)
{
    pImgData = new ImageData();

#define COPY_PARAMETER(p) pImgData->p = p
    COPY_PARAMETER(m_eDataType);
    COPY_PARAMETER(m_uiChannelNum);
    COPY_PARAMETER(m_fSlope);
    COPY_PARAMETER(m_fIntercept);
    COPY_PARAMETER(m_vImgOrientation[0]);
    COPY_PARAMETER(m_vImgOrientation[1]);
    COPY_PARAMETER(m_vImgOrientation[2]);
    COPY_PARAMETER(m_ptImgPositon);
    COPY_PARAMETER(m_uiDim[0]);
    COPY_PARAMETER(m_uiDim[1]);
    COPY_PARAMETER(m_uiDim[2]);
    COPY_PARAMETER(m_dSpacing[0]);
    COPY_PARAMETER(m_dSpacing[1]);
    COPY_PARAMETER(m_dSpacing[2]);
    COPY_PARAMETER(m_fMinScalar);
    COPY_PARAMETER(m_fMaxScalar);
    COPY_PARAMETER(m_bCalcMinMax);
#undef COPY_PARAMETER
}

void ImageData::DeepCopy(ImageData *&pImgData)
{
    this->ShallowCopy(pImgData);

    //Copy this image data
    pImgData->AllocateMemory();
    const size_t imemSize = this->GetVolumeDataSize_i();
    memcpy(pImgData->m_pMappedData.get(), this->m_pMappedData.get(), imemSize );
}

void ImageData::CalculateMinMax_i()
{
    void* pMappedData = m_pMappedData.get();
    if (nullptr == pMappedData)
    {
        throw std::exception("Volume data is null!");
    }

    switch(m_eDataType)
    {
    case CHAR: 
        this->FindMinMax_i((char *)pMappedData);
        break;
    case UCHAR: 
        this->FindMinMax_i((unsigned char *)pMappedData);
        break;
    case USHORT:
        this->FindMinMax_i((unsigned short *)pMappedData);
        break;
    case SHORT:
        this->FindMinMax_i((short *)pMappedData);
        break;
    case FLOAT:
        this->FindMinMax_i((float *)pMappedData);
        break;
    default:
        IO_THROW_EXCEPTION("Undefined image type!");
    }

    m_bCalcMinMax = true;
}

size_t ImageData::GetVolumeDataSize_i()
{
    size_t imemSize = m_uiDim[0] * m_uiDim[1] * m_uiDim[2] * m_uiChannelNum;
    switch(m_eDataType)
    {
    case CHAR:
        imemSize *= sizeof(char);
        break;
    case UCHAR:
        imemSize *= sizeof(unsigned char);
        break;
    case USHORT:
        imemSize *= sizeof(unsigned short);
        break;
    case SHORT:
        imemSize *=sizeof(short);
        break;
    case FLOAT:
        imemSize *= sizeof(float);
        break;
    default:
        IO_THROW_EXCEPTION("Undefined image type!");
    }

    return imemSize;
}


MED_IMAGING_END_NAMESPACE