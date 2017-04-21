#include "mi_ray_caster.h"

#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_texture_1d_array.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster_inner_buffer.h"
#include "mi_ray_casting_cpu.h"
#include "mi_ray_casting_gpu.h"
#include "mi_ray_casting_cpu_brick_acc.h"
#include "mi_ray_caster_canvas.h"

MED_IMAGING_BEGIN_NAMESPACE

RayCaster::RayCaster():m_pInnerBuffer(new RayCasterInnerBuffer()),
m_matVolume2World(Matrix4::S_IDENTITY_MATRIX),
m_fSampleRate(0.5f),
m_fGlobalWW(1.0f),
m_fGlobalWL(0.0f),
m_pPseudoColorArray(nullptr),
m_uiPseudoColorLength(256),
m_fSSDGray(1.0f),
m_bJitteringEnabled(false),
m_vBoundingMin(Vector3f(0,0,0)),
m_vBoundingMax(Vector3f(32,32,32)),
m_eMaskMode(MASK_NONE),
m_eCompositeMode(COMPOSITE_AVERAGE),
m_eInterpolationMode(LINEAR),
m_eShadingMode(SHADING_NONE),
m_eColorInverseMode(COLOR_INVERSE_DISABLE),
m_eStrategy(CPU_BASE),
m_pBrickCorner(nullptr),
m_pVolumeBrickUnit(nullptr),
m_pMaskBrickUnit(nullptr),
m_pVolumeBrickInfo(nullptr),
m_pMaskBrickInfo(nullptr),
m_uiBrickSize(32),
m_uiBrickExpand(2)
{

}

void RayCaster::initialize()
{
    
}

void RayCaster::finialize()
{
    m_pInnerBuffer->release_buffer();
}

RayCaster::~RayCaster()
{

}

void RayCaster::render(int iTestCode)
{
    //clock_t t0 = clock();

    if (CPU_BASE == m_eStrategy)
    {
        if (!m_pRayCastingCPU)
        {
            m_pRayCastingCPU.reset(new RayCastingCPU(shared_from_this()));
        }
        m_pRayCastingCPU->render(iTestCode);
    }
    else if (CPU_BRICK_ACCELERATE == m_eStrategy)
    {
        if (!m_pRayCastingCPUBrickAcc)
        {
            m_pRayCastingCPUBrickAcc.reset(new RayCastingCPUBrickAcc(shared_from_this()));
        }
        m_pRayCastingCPUBrickAcc->render(iTestCode);
    }
    else if (GPU_BASE == m_eStrategy)
    {
        if (!m_pRayCastingGPU)
        {
            m_pRayCastingGPU.reset(new RayCastingGPU(shared_from_this()));
        }

        {
            CHECK_GL_ERROR;

            FBOStack fboStack;
            m_pCanvas->get_fbo()->bind();

            CHECK_GL_ERROR;

            glDrawBuffer(GL_COLOR_ATTACHMENT0);

            CHECK_GL_ERROR;

            //Clear 
            glClearColor(0,0,0,0);
            glClearDepth(1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            //Viewport
            int iWidth , iHeight;
            m_pCanvas->get_display_size(iWidth , iHeight);
            glViewport(0,0,iWidth , iHeight);

            CHECK_GL_ERROR;

            m_pRayCastingGPU->render(iTestCode);
        }
    }

    //clock_t t1 = clock();
    //std::cout << "<<<>>><<<>>><<<>>><<<>>><<<>>>\n";
    //std::cout << "Ray casting cost : " << double(t1-t0) << "ms\n";
    //std::cout << "<<<>>><<<>>><<<>>><<<>>><<<>>>\n";
}

void RayCaster::set_volume_data(std::shared_ptr<ImageData> image_data)
{
    m_pVolumeData = image_data;
}

void RayCaster::set_mask_data(std::shared_ptr<ImageData> image_data)
{
    m_pMaskData = image_data;
}

void RayCaster::set_volume_data_texture(std::vector<GLTexture3DPtr> vecTex)
{
    m_vecVolumeDataTex = vecTex;
}

void RayCaster::set_mask_data_texture(std::vector<GLTexture3DPtr> vecTex)
{
    m_vecMaskDataTex =vecTex;
}

void RayCaster::set_entry_exit_points(std::shared_ptr<EntryExitPoints> pEEPs)
{
    m_pEntryExitPoints = pEEPs;
}

void RayCaster::set_camera(std::shared_ptr<CameraBase> pCamera)
{
    m_pCamera = pCamera;
}

void RayCaster::set_volume_to_world_matrix(const Matrix4& mat)
{
    m_matVolume2World = mat;
}

void RayCaster::set_sample_rate(float fSampleRate)
{
    m_fSampleRate = fSampleRate;
}

void RayCaster::set_mask_label_level(LabelLevel eLabelLevel)
{
    m_pInnerBuffer->set_mask_label_level(eLabelLevel);
}

void RayCaster::set_visible_labels(std::vector<unsigned char> vecLabels)
{
    m_pInnerBuffer->set_visible_labels(vecLabels);
}

void RayCaster::set_window_level(float fWW , float fWL , unsigned char ucLabel)
{
    m_pInnerBuffer->set_window_level(fWW , fWL , ucLabel);
}

void RayCaster::set_global_window_level(float fWW , float fWL)
{
    m_fGlobalWW = fWW;
    m_fGlobalWL = fWL;
}

void RayCaster::set_pseudo_color_texture(GLTexture1DPtr pTex , unsigned int uiLength)
{
    m_pPseudoColorTexture = pTex;
    m_uiPseudoColorLength= uiLength;
}

GLTexture1DPtr RayCaster::get_pseudo_color_texture(unsigned int& uiLength) const
{
    uiLength = m_uiPseudoColorLength;
    return m_pPseudoColorTexture;
}

void RayCaster::set_pseudo_color_array(unsigned char* pArray , unsigned int uiLength)
{
    m_pPseudoColorArray = pArray;
    m_uiPseudoColorLength = uiLength;
}

void RayCaster::set_transfer_function_texture(GLTexture1DArrayPtr pTexArray)
{
    m_pTransferFunction = pTexArray;
}

void RayCaster::set_sillhouette_enhancement()
{

}

void RayCaster::set_boundary_enhancement()
{

}

void RayCaster::set_material()
{

}

void RayCaster::set_light_color()
{

}

void RayCaster::set_light_factor()
{

}

void RayCaster::set_ssd_gray(float fGray)
{
    m_fSSDGray = fGray;
}

void RayCaster::set_jittering_enabled(bool flag)
{
    m_bJitteringEnabled = flag;
}

void RayCaster::set_bounding(const Vector3f& vMin, const Vector3f& vMax)
{
    m_vBoundingMin = vMin;
    m_vBoundingMax = vMax;
}

void RayCaster::set_clipping_plane_function(const std::vector<Vector4f> &vecFunc)
{
    m_vClippingPlaneFunc = vecFunc;
}

void RayCaster::set_mask_mode(MaskMode eMode)
{
    m_eMaskMode = eMode;
}

void RayCaster::set_composite_mode(CompositeMode eMode)
{
    m_eCompositeMode = eMode;
}

void RayCaster::set_interpolation_mode(InterpolationMode eMode)
{
    m_eInterpolationMode = eMode;
}

void RayCaster::set_shading_mode(ShadingMode eMode)
{
    m_eShadingMode = eMode;
}

void RayCaster::set_color_inverse_mode(ColorInverseMode eMode)
{
    m_eColorInverseMode = eMode;
}

void RayCaster::set_canvas(std::shared_ptr<RayCasterCanvas> pCanvas)
{
    m_pCanvas = pCanvas;
}

void RayCaster::set_strategy(RayCastingStrategy eS)
{
    m_eStrategy = eS;
}

void RayCaster::set_brick_size(unsigned int uiBrickSize)
{
    m_uiBrickSize = uiBrickSize;
}

void RayCaster::set_brick_expand(unsigned int uiBrickExpand)
{
    m_uiBrickExpand = uiBrickExpand;
}

void RayCaster::set_brick_corner(BrickCorner* pBC)
{
    m_pBrickCorner = pBC;
}

void RayCaster::set_volume_brick_unit(BrickUnit* pBU)
{
    m_pVolumeBrickUnit = pBU;
}

void RayCaster::set_mask_brick_unit(BrickUnit* pBU)
{
    m_pMaskBrickUnit = pBU;
}

void RayCaster::set_mask_brick_info(MaskBrickInfo* pMBI)
{
    m_pMaskBrickInfo = pMBI;
}

void RayCaster::set_volume_brick_info(VolumeBrickInfo* pVBI)
{
    m_pVolumeBrickInfo = pVBI;
}

const std::vector<BrickDistance>& RayCaster::get_brick_distance() const
{
    return m_pRayCastingCPUBrickAcc->get_brick_distance();
}

unsigned int RayCaster::get_ray_casting_brick_count() const
{
    return m_pRayCastingCPUBrickAcc->get_ray_casting_brick_count();
}

std::shared_ptr<ImageData> RayCaster::get_volume_data()
{
    return m_pVolumeData;
}

std::vector<GLTexture3DPtr> RayCaster::get_volume_data_texture()
{
    return m_vecVolumeDataTex;
}

float RayCaster::get_sample_rate() const
{
    return m_fSampleRate;
}

void RayCaster::get_global_window_level(float& fWW , float& fWL) const
{
    fWW = m_fGlobalWW;
    fWL = m_fGlobalWL;
}

std::shared_ptr<EntryExitPoints> RayCaster::get_entry_exit_points() const
{
    return m_pEntryExitPoints;
}



MED_IMAGING_END_NAMESPACE