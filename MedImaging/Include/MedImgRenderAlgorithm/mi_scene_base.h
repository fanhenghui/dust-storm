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
    void set_name(const std::string& sName);
    const std::string& get_name() const;

    virtual ~SceneBase();
    virtual void initialize();
    virtual void finalize();
    virtual void set_display_size(int iWidth , int iHeight);
    void get_display_size(int& iWidth, int& iHeight) const;
    virtual void render(int iTestCode);

    virtual void rotate(const Point2& ptPre , const Point2& ptCur);
    virtual void zoom(const Point2& ptPre , const Point2& ptCur);
    virtual void pan(const Point2& ptPre , const Point2& ptCur);

    std::shared_ptr<CameraBase> get_camera();
    void render_to_back();

    void set_dirty(bool bFlag);
    bool get_dirty() const;

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