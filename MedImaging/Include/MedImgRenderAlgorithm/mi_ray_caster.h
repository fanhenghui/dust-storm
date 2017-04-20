#ifndef MED_IMAGING_RAY_CASTER_H_
#define MED_IMAGING_RAY_CASTER_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_vector4f.h"

#include "MedImgGLResource/mi_gl_resource_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class EntryExitPoints;
class ImageData;
class CameraBase;
class RayCasterInnerBuffer;
class RayCasterCanvas;

class RenderAlgo_Export RayCaster : public std::enable_shared_from_this<RayCaster>
{
    friend class RayCastingCPU;
    friend class RayCastingCPUBrickAcc;
    friend class RayCastingGPU;

public:
    RayCaster();

    ~RayCaster();

    void initialize();

    void finialize();

    void render(int iTestCode = 0);

    //Ray casting strategy
    void set_strategy(RayCastingStrategy eS);

    void set_canvas(std::shared_ptr<RayCasterCanvas> pCanvas);

    //Mask label level
    //Default is L_8
    void set_mask_label_level(LabelLevel eLabelLevel);

    //////////////////////////////////////////////////////////////////////////
    //Input data
    //////////////////////////////////////////////////////////////////////////

    //Volume & mask texture/array
    void set_volume_data(std::shared_ptr<ImageData> pImgData);
    void set_mask_data(std::shared_ptr<ImageData> pImgData);
    void set_volume_data_texture(std::vector<GLTexture3DPtr> vecTex);
    void set_mask_data_texture(std::vector<GLTexture3DPtr> vecTex);

    std::shared_ptr<ImageData> get_volume_data();
    std::vector<GLTexture3DPtr> get_volume_data_texture( );

    //Brick acceleration
    void set_brick_size(unsigned int uiBrickSize);
    void set_brick_expand(unsigned int uiBrickExpand);
    void set_brick_corner(BrickCorner* pBC);
    void set_volume_brick_unit(BrickUnit* pBU);
    void set_mask_brick_unit(BrickUnit* pBU);
    void set_volume_brick_info(VolumeBrickInfo* pVBI);
    void set_mask_brick_info(MaskBrickInfo* pMBI);//Here don't need label visibility status , just set current info

    //Entry exit points
    void set_entry_exit_points(std::shared_ptr<EntryExitPoints> pEEPs);

    std::shared_ptr<EntryExitPoints> get_entry_exit_points() const;


    //////////////////////////////////////////////////////////////////////////
    //Ray casting parameter
    //////////////////////////////////////////////////////////////////////////

    //Volume modeling parameter
    void set_camera(std::shared_ptr<CameraBase> pCamera);
    void set_volume_to_world_matrix(const Matrix4& mat);

    //Sample rate
    void set_sample_rate(float fSampleRate);

    float get_sample_rate() const;

    //Label parameter
    void set_visible_labels(std::vector<unsigned char> vecLabels);

    //Window level parameter
    //Here
    void set_window_level(float fWW , float fWL , unsigned char ucLabel);
    void set_global_window_level(float fWW , float fWL);

    void get_global_window_level(float& fWW , float& fWL) const;

    //Transfer function & pseudo color parameter
    void set_pseudo_color_texture(GLTexture1DPtr pTex , unsigned int uiLength);
    GLTexture1DPtr get_pseudo_color_texture(unsigned int& uiLength) const;

    //RGB8 array
    void set_pseudo_color_array(unsigned char* pArray , unsigned int uiLength);
    void set_transfer_function_texture(GLTexture1DArrayPtr pTexArray);

    //Enhancement parameter
    void set_sillhouette_enhancement();
    void set_boundary_enhancement();

    //Shading parameter
    void set_material();
    void set_light_color();
    void set_light_factor();

    //SSD gray value
    void set_ssd_gray(float fGray);

    //Jittering to prevent wooden artifacts
    void set_jittering_enabled(bool bFlag);

    //Bounding box
    void set_bounding(const Vector3f& vMin, const Vector3f& vMax);

    //Clipping plane
    void set_clipping_plane_function(const std::vector<Vector4f> &vecFunc);

    //Ray casting mode parameter
    void set_mask_mode(MaskMode eMode);
    void set_composite_mode(CompositeMode eMode);
    void set_interpolation_mode(InterpolationMode eMode);
    void set_shading_mode(ShadingMode eMode);
    void set_color_inverse_mode(ColorInverseMode eMode);


    //////////////////////////////////////////////////////////////////////////
    //For testing
    const std::vector<BrickDistance>& get_brick_distance() const;
    unsigned int get_ray_casting_brick_count() const;
protected:
    //Input data
    std::shared_ptr<ImageData> m_pVolumeData;
    std::vector<GLTexture3DPtr> m_vecVolumeDataTex;

    std::shared_ptr<ImageData> m_pMaskData;
    std::vector<GLTexture3DPtr> m_vecMaskDataTex;

    //Brick acceleration
    BrickCorner* m_pBrickCorner;
    BrickUnit* m_pVolumeBrickUnit;
    BrickUnit* m_pMaskBrickUnit;
    VolumeBrickInfo* m_pVolumeBrickInfo;
    MaskBrickInfo* m_pMaskBrickInfo;
    unsigned int m_uiBrickSize;
    unsigned int m_uiBrickExpand;

    //Entry exit points
    std::shared_ptr<EntryExitPoints> m_pEntryExitPoints;

    std::shared_ptr<CameraBase> m_pCamera;

    Matrix4 m_matVolume2World;

    //Data sample rate(DVR 0.5 , MIPs 1.0) 
    float m_fSampleRate;

    //Global window level for MIPs mode
    float m_fGlobalWW;
    float m_fGlobalWL;

    //Transfer function & pseudo color 
    GLTexture1DArrayPtr m_pTransferFunction;
    GLTexture1DPtr m_pPseudoColorTexture;
    unsigned char* m_pPseudoColorArray;
    unsigned int m_uiPseudoColorLength;

    //Inner buffer to contain label based parameter
    std::shared_ptr<RayCasterInnerBuffer> m_pInnerBuffer;

    //SSD gray value
    float m_fSSDGray;

    //DVR jittering flag
    bool m_bJitteringEnabled;

    //Bounding
    Vector3f m_vBoundingMin;
    Vector3f m_vBoundingMax;

    //Clipping plane
    std::vector<Vector4f> m_vClippingPlaneFunc;

    //Ray casting mode
    MaskMode m_eMaskMode;
    CompositeMode m_eCompositeMode;
    InterpolationMode m_eInterpolationMode;
    ShadingMode m_eShadingMode;
    ColorInverseMode m_eColorInverseMode;

    //Processing unit type
    RayCastingStrategy m_eStrategy;

    std::shared_ptr<RayCastingCPU> m_pRayCastingCPU;
    std::shared_ptr<RayCastingGPU> m_pRayCastingGPU;
    std::shared_ptr<RayCastingCPUBrickAcc> m_pRayCastingCPUBrickAcc;

    //Canvas for rendering
    std::shared_ptr<RayCasterCanvas> m_pCanvas;
};

MED_IMAGING_END_NAMESPACE


#endif