#ifndef MED_IMAGING_RENDER_ENTRY_EXIT_POINTS_H
#define MED_IMAGING_RENDER_ENTRY_EXIT_POINTS_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"
#include "MedImgGLResource/mi_gl_object.h"
#include "MedImgArithmetic/mi_vector4f.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLTexture2D;
class CameraBase;
class ImageData;
class CameraCalculator;
class RenderAlgo_Export EntryExitPoints
{
public:
    EntryExitPoints();

    virtual ~EntryExitPoints();

    void SetDisplaySize(int iWidth , int iHeight);

    void GetDisplaySize(int& iWidth , int& iHeight);

    void SetStrategy( RayCastingStrategy eStrategy );

    virtual void Initialize();

    virtual void Finialize();

    std::shared_ptr<GLTexture2D> GetEntryPointsTexture();

    std::shared_ptr<GLTexture2D> GetExitPointsTexture();

    Vector4f* GetEntryPointsArray();

    Vector4f* GetExitPointsArray();

    void SetImageData(std::shared_ptr<ImageData> pImgData);

    void SetCamera(std::shared_ptr<CameraBase> pCamera);

    void SetCameraCalculator(std::shared_ptr<CameraCalculator> pCameraCal);

    virtual void CalculateEntryExitPoints() = 0;

public:
    void DebugOutputEntryPoints(const std::string& sFileName);

    void DebugOutputExitPoints(const std::string& sFileName);

protected:
    std::shared_ptr<GLTexture2D> m_pEntryTex;
    std::shared_ptr<GLTexture2D> m_pExitTex;
    std::unique_ptr<Vector4f[]> m_pEntryBuffer;
    std::unique_ptr<Vector4f[]> m_pExitBuffer;
    int m_iWidth;
    int m_iHeight;
    std::shared_ptr<CameraBase> m_pCamera;
    std::shared_ptr<ImageData> m_pImgData;
    std::shared_ptr<CameraCalculator> m_pCameraCalculator;
    bool m_bInit;

    RayCastingStrategy m_eStrategy;
};

MED_IMAGING_END_NAMESPACE

#endif