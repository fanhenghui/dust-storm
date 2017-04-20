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

bool ImageData::mem_allocate()
{
    const size_t imemSize = get_data_size_i();
    m_pMappedData.reset(new char[imemSize]);
    m_bCalcMinMax = false;
    return true;
}

float ImageData::get_min_scalar()
{
    if(!m_bCalcMinMax)
    {
        find_min_max_i();
    }
    return m_fMinScalar;
}

float ImageData::get_max_scalar()
{
    if(!m_bCalcMinMax)
    {
        find_min_max_i();
    }
    return m_fMaxScalar;
}

bool ImageData::regulate_wl(float& fWindow, float& fLevel)
{
    // CT should apply slope and intercept
    // MR has always slope(1) and intercept(0)
    if (m_fSlope < DOUBLE_EPSILON)
    {
        return false;
    }

    double dMin = get_min_scalar();

    fLevel = (fLevel - m_fIntercept)/m_fSlope;
    fWindow = fWindow/m_fSlope;

    return true;
}


void ImageData::normalize_wl(float& fWindow, float& fLevel)
{
    const static float f65535R = 1.0f/65535.0f;
    const static float f255R = 1.0f/255.0f;

    float fMin = get_min_scalar();

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


bool ImageData::regulate_normalize_wl(float& fWindow, float& fLevel)
{
    if (!regulate_wl(fWindow , fLevel))
    {
        return false;
    }

    normalize_wl(fWindow , fLevel);

    return true;
}

void ImageData::get_pixel_value(unsigned int x,unsigned int y ,unsigned int z , double& pValue) const
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

void ImageData::get_pixel_value(const Point3& ptPos , double& pValue) const
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
            dPixelValue =(double) sampler.sample_3d_linear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (char*)pMappedData);
            break;
        }
    case UCHAR:
        {
            Sampler<unsigned char> sampler;
            dPixelValue =(double) sampler.sample_3d_linear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (unsigned char*)pMappedData);
            break;
        }
    case USHORT:
        {
            Sampler<unsigned short> sampler;
            dPixelValue =(double) sampler.sample_3d_linear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (unsigned short*)pMappedData);
            break;
        }
    case SHORT:
        {
            Sampler<short> sampler;
            dPixelValue =(double) sampler.sample_3d_linear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (short*)pMappedData);
            break;
        }
    case FLOAT:
        {
            Sampler<float> sampler;
            dPixelValue =(float) sampler.sample_3d_linear((float)ptPos.x , (float)ptPos.y , (float)ptPos.z , 
                m_uiDim[0] , m_uiDim[1], m_uiDim[2] , (float*)pMappedData);
            break;
        }
    default:
        IO_THROW_EXCEPTION("Undefined image type!");
    }
    pValue = dPixelValue;
}

void ImageData::set_data_dirty()
{
    m_bCalcMinMax = false;
}

void* ImageData::get_pixel_pointer()
{
    return m_pMappedData.get();
}

void ImageData::shallow_copy(ImageData *&pImgData)
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

void ImageData::deep_copy(ImageData *&pImgData)
{
    this->shallow_copy(pImgData);

    //Copy this image data
    pImgData->mem_allocate();
    const size_t imemSize = this->get_data_size_i();
    memcpy(pImgData->m_pMappedData.get(), this->m_pMappedData.get(), imemSize );
}

void ImageData::find_min_max_i()
{
    void* pMappedData = m_pMappedData.get();
    if (nullptr == pMappedData)
    {
        throw std::exception("Volume data is null!");
    }

    switch(m_eDataType)
    {
    case CHAR: 
        this->find_min_max_i((char *)pMappedData);
        break;
    case UCHAR: 
        this->find_min_max_i((unsigned char *)pMappedData);
        break;
    case USHORT:
        this->find_min_max_i((unsigned short *)pMappedData);
        break;
    case SHORT:
        this->find_min_max_i((short *)pMappedData);
        break;
    case FLOAT:
        this->find_min_max_i((float *)pMappedData);
        break;
    default:
        IO_THROW_EXCEPTION("Undefined image type!");
    }

    m_bCalcMinMax = true;
}

size_t ImageData::get_data_size_i()
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