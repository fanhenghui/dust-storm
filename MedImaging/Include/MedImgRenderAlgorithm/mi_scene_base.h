#ifndef MED_IMAGING_SCENE_BASE_H_H
#define MED_IMAGING_SCENE_BASE_H_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_resource_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export SceneBase 
{
public:
    SceneBase();
    SceneBase(int iWidth , int iHeight);
    void SetName(const std::string& sName);
    const std::string& GetName() const;

    virtual ~SceneBase();
    virtual void Initialize();
    virtual void Finalize();
    virtual void SetDisplaySize(int iWidth , int iHeight);
    void GetDisplaySize(int& iWidth, int& iHeight) const;
    virtual void Render(int iTestCode);

    virtual void Rotate(const Point2& ptPre , const Point2& ptCur);
    virtual void Zoom(const Point2& ptPre , const Point2& ptCur);
    virtual void Pan(const Point2& ptPre , const Point2& ptCur);

    std::shared_ptr<CameraBase> GetCamera();
    void RenderToBack();

    void SetDirty(bool bFlag);
    bool GetDirty() const;

protected:
    int m_iWidth , m_iHeight;
    GLFBOPtr m_pSceneFBO;
    GLTexture2DPtr m_pSceneColorAttach0;
    GLTexture2DPtr m_pSceneDepthAttach;
    std::shared_ptr<CameraBase> m_pCamera;
    bool m_bDirty;
    std::string m_sName;

};

MED_IMAGING_END_NAMESPACE
#endif