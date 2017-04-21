#ifndef MED_IMAGING_SCENE_BASE_H_H
#define MED_IMAGING_SCENE_BASE_H_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_resource_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export SceneBase 
{
public:
    SceneBase();
    SceneBase(int width , int height);
    void set_name(const std::string& sName);
    const std::string& get_name() const;

    virtual ~SceneBase();
    virtual void initialize();
    virtual void finalize();
    virtual void set_display_size(int width , int height);
    void get_display_size(int& width, int& height) const;
    virtual void render(int test_code);

    virtual void rotate(const Point2& pre_pt , const Point2& cur_pt);
    virtual void zoom(const Point2& pre_pt , const Point2& cur_pt);
    virtual void pan(const Point2& pre_pt , const Point2& cur_pt);

    std::shared_ptr<CameraBase> get_camera();
    void render_to_back();

    void set_dirty(bool flag);
    bool get_dirty() const;

protected:
    int _width , _height;
    GLFBOPtr m_pSceneFBO;
    GLTexture2DPtr m_pSceneColorAttach0;
    GLTexture2DPtr m_pSceneDepthAttach;
    std::shared_ptr<CameraBase> _camera;
    bool m_bDirty;
    std::string m_sName;

};

MED_IMAGING_END_NAMESPACE
#endif