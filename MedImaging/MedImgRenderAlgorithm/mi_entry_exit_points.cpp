#include "mi_entry_exit_points.h"

#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgArithmetic/mi_camera_base.h"

#include "mi_camera_calculator.h"

MED_IMAGING_BEGIN_NAMESPACE

EntryExitPoints::EntryExitPoints():_width(4),_height(4),m_bInit(false),m_eStrategy(CPU_BASE)
{
    m_pEntryBuffer.reset(new Vector4f[_width*_height]);
    m_pExitBuffer.reset(new Vector4f[_width*_height]);
    UIDType uid;
    m_pEntryTex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(uid);
    m_pExitTex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object(uid);
}

void EntryExitPoints::initialize()
{
    if (!m_bInit)
    {
        m_pEntryTex->initialize();
        m_pExitTex->initialize();
        m_bInit = true;
    }
}

void EntryExitPoints::finialize()
{
    if (m_bInit)
    {
        GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(m_pEntryTex->get_uid());
        GLResourceManagerContainer::instance()->get_texture_2d_manager()->remove_object(m_pExitTex->get_uid());
        GLResourceManagerContainer::instance()->get_texture_2d_manager()->update();
        m_bInit = false;
    }
}

EntryExitPoints::~EntryExitPoints()
{

}

void EntryExitPoints::set_display_size(int iWidth , int iHeight)
{
    _width = iWidth;
    _height = iHeight;
    m_pEntryBuffer.reset(new Vector4f[_width*_height]);
    m_pExitBuffer.reset(new Vector4f[_width*_height]);

    //resize texture
    if (GPU_BASE == m_eStrategy)
    { 
        initialize();

        CHECK_GL_ERROR;

        m_pEntryTex->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        m_pEntryTex->load(GL_RGBA32F , _width , _height , GL_RGBA , GL_FLOAT , NULL);
        m_pEntryTex->unbind();

        m_pExitTex->bind();
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        m_pExitTex->load(GL_RGBA32F , _width , _height , GL_RGBA , GL_FLOAT , NULL);
        m_pExitTex->unbind();

        CHECK_GL_ERROR;
    }
}

void EntryExitPoints::get_display_size(int& iWidth , int& iHeight)
{
    iWidth = _width;
    iHeight = _height;
}

std::shared_ptr<GLTexture2D> EntryExitPoints::get_entry_points_texture()
{
    return m_pEntryTex;
}

std::shared_ptr<GLTexture2D> EntryExitPoints::get_exit_points_texture()
{
    return m_pExitTex;
}

Vector4f* EntryExitPoints::get_entry_points_array()
{
    return m_pEntryBuffer.get();
}

Vector4f* EntryExitPoints::get_exit_points_array()
{
    return m_pExitBuffer.get();
}

void EntryExitPoints::set_image_data(std::shared_ptr<ImageData> image_data)
{
    m_pImgData = image_data;
}

void EntryExitPoints::set_camera(std::shared_ptr<CameraBase> pCamera)
{
    m_pCamera = pCamera;
}

void EntryExitPoints::set_camera_calculator(std::shared_ptr<CameraCalculator> pCameraCal)
{
    m_pCameraCalculator = pCameraCal;
}

void EntryExitPoints::debug_output_entry_points(const std::string& sFileName)
{
    Vector4f* pPoints = m_pEntryBuffer.get();
    std::ofstream out(sFileName , std::ios::binary | std::ios::out);
    if (out.is_open())
    {
        std::unique_ptr<unsigned char[]> pRGB(new unsigned char[_width*_height*3]);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pImgData);
        unsigned int *uiDim = m_pImgData->_dim;
        float fDimR[3] = { 1.0f/(float)uiDim[0],1.0f/(float)uiDim[1],1.0f/(float)uiDim[2]};
        unsigned char r,g,b;
        float fR , fG , fB;
        for (int i = 0 ; i < _width*_height ; ++i)
        {
            fR =pPoints[i]._m[0] *fDimR[0]*255.0f;
            fG =pPoints[i]._m[1] *fDimR[1]*255.0f;
            fB =pPoints[i]._m[2] *fDimR[2]*255.0f;

            fR = fR > 255.0f ? 255.0f : fR;
            fR = fR <0.0f ? 0.0f : fR;

            fG = fG > 255.0f ? 255.0f : fG;
            fG = fG <0.0f ? 0.0f : fG;

            fB = fB > 255.0f ? 255.0f : fB;
            fB = fB <0.0f ? 0.0f : fB;

            r = unsigned char(fR);
            g = unsigned char(fG);
            b = unsigned char(fB);

            pRGB[i*3] = r;
            pRGB[i*3+1] = g;
            pRGB[i*3+2] = b;

        }

        out.write((char*)pRGB.get()  , _width*_height*3);
        out.close();
    }
    else
    {
        //TODO LOG
        std::cout << "Open file " << sFileName << " failed!\n";
    }
}

void EntryExitPoints::debug_output_exit_points(const std::string& sFileName)
{
    Vector4f* pPoints = m_pExitBuffer.get();
    std::ofstream out(sFileName , std::ios::binary | std::ios::out);
    if (out.is_open())
    {
        std::unique_ptr<unsigned char[]> pRGB(new unsigned char[_width*_height*3]);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pImgData);
        unsigned int *uiDim = m_pImgData->_dim;
        float fDimR[3] = { 1.0f/(float)uiDim[0],1.0f/(float)uiDim[1],1.0f/(float)uiDim[2]};
        unsigned char r,g,b;
        float fR , fG , fB;
        for (int i = 0 ; i < _width*_height ; ++i)
        {
            fR =pPoints[i]._m[0] *fDimR[0]*255.0f;
            fG =pPoints[i]._m[1] *fDimR[1]*255.0f;
            fB =pPoints[i]._m[2] *fDimR[2]*255.0f;

            fR = fR > 255.0f ? 255.0f : fR;
            fR = fR <0.0f ? 0.0f : fR;

            fG = fG > 255.0f ? 255.0f : fG;
            fG = fG <0.0f ? 0.0f : fG;

            fB = fB > 255.0f ? 255.0f : fB;
            fB = fB <0.0f ? 0.0f : fB;

            r = unsigned char(fR);
            g = unsigned char(fG);
            b = unsigned char(fB);

            pRGB[i*3] = r;
            pRGB[i*3+1] = g;
            pRGB[i*3+2] = b;

        }

        out.write((char*)pRGB.get()  , _width*_height*3);
        out.close();
    }
    else
    {
        //TODO LOG
        std::cout << "Open file " << sFileName << " failed!\n";
    }
}

void EntryExitPoints::set_strategy( RayCastingStrategy eStrategy )
{
    m_eStrategy = eStrategy;
}



MED_IMAGING_END_NAMESPACE