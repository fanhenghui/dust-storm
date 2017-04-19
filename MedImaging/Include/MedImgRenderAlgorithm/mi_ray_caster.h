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

    void Initialize();

    void Finialize();

    void Render(int iTestCode = 0);

    //Ray casting strategy
    void SetStrategy(RayCastingStrategy eS);

    void SetCanvas(std::shared_ptr<RayCasterCanvas> pCanvas);

    //Mask label level
    //Default is L_8
    void SetMaskLabelLevel(LabelLevel eLabelLevel);

    //////////////////////////////////////////////////////////////////////////
    //Input data
    //////////////////////////////////////////////////////////////////////////

    //Volume & mask texture/array
    void SetVolumeData(std::shared_ptr<ImageData> pImgData);
    void SetMaskData(std::shared_ptr<ImageData> pImgData);
    void SetVolumeDataTexture(std::vector<GLTexture3DPtr> vecTex);
    void SetMaskDataTexture(std::vector<GLTexture3DPtr> vecTex);

    std::shared_ptr<ImageData> GetVolumeData();
    std::vector<GLTexture3DPtr> GetVolumeDataTexture( );

    //Brick acceleration
    void SetBrickSize(unsigned int uiBrickSize);
    void SetBrickExpand(unsigned int uiBrickExpand);
    void SetBrickCorner(BrickCorner* pBC);
    void SetVolumeBrickUnit(BrickUnit* pBU);
    void SetMaskBrickUnit(BrickUnit* pBU);
    void SetVolumeBrickInfo(VolumeBrickInfo* pVBI);
    void SetMaskBrickInfo(MaskBrickInfo* pMBI);//Here don't need label visibility status , just set current info

    //Entry exit points
    void SetEntryExitPoints(std::shared_ptr<EntryExitPoints> pEEPs);

    std::shared_ptr<EntryExitPoints> GetEntryExitPoints() const;


    //////////////////////////////////////////////////////////////////////////
    //Ray casting parameter
    //////////////////////////////////////////////////////////////////////////

    //Volume modeling parameter
    void SetCamera(std::shared_ptr<CameraBase> pCamera);
    void SetVolumeToWorldMatrix(const Matrix4& mat);

    //Sample rate
    void SetSampleRate(float fSampleRate);

    float GetSampleRate() const;

    //Label parameter
    void SetVisibleLabels(std::vector<unsigned char> vecLabels);

    //Window level parameter
    //Here
    void SetWindowlevel(float fWW , float fWL , unsigned char ucLabel);
    void SetGlobalWindowLevel(float fWW , float fWL);

    void GetGlobalWindowLevel(float& fWW , float& fWL) const;

    //Transfer function & pseudo color parameter
    void SetPseudoColorTexture(GLTexture1DPtr pTex , unsigned int uiLength);
    GLTexture1DPtr GetPseudoColorTexture(unsigned int& uiLength) const;

    //RGB8 array
    void SetPseudoColorArray(unsigned char* pArray , unsigned int uiLength);
    void SetTransferFunctionTexture(GLTexture1DArrayPtr pTexArray);

    //Enhancement parameter
    void SetSilhouetteEnhancement();
    void SetBoundaryEnhancement();

    //Shading parameter
    void SetMaterial();
    void SetLightColor();
    void SetLightFactor();

    //SSD gray value
    void SetSSDGray(float fGray);

    //Jittering to prevent wooden artifacts
    void SetJitteringEnabled(bool bFlag);

    //Bounding box
    void SetBounding(const Vector3f& vMin, const Vector3f& vMax);

    //Clipping plane
    void SetClippingPlaneFunction(const std::vector<Vector4f> &vecFunc);

    //Ray casting mode parameter
    void SetMaskMode(MaskMode eMode);
    void SetCompositeMode(CompositeMode eMode);
    void SetInterpolationMode(InterpolationMode eMode);
    void SetShadingMode(ShadingMode eMode);
    void SetColorInverseMode(ColorInverseMode eMode);


    //////////////////////////////////////////////////////////////////////////
    //For testing
    const std::vector<BrickDistance>& GetBrickDistance() const;
    unsigned int GetRayCastingBrickCount() const;
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