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
    virtual ~SceneBase();

    void set_name(const std::string& name);
    const std::string& get_name() const;

    virtual void initialize();
    virtual void finalize();

    virtual void set_display_size(int width , int height);
    void get_display_size(int& width, int& height) const;

    virtual void rotate(const Point2& pre_pt , const Point2& cur_pt);
    virtual void zoom(const Point2& pre_pt , const Point2& cur_pt);
    virtual void pan(const Point2& pre_pt , const Point2& cur_pt);

    std::shared_ptr<CameraBase> get_camera();

    virtual void render(int test_code);
    void render_to_back();

    void set_dirty(bool flag);
    bool get_dirty() const;

protected:
    int _width , _height;

    GLFBOPtr _scene_fbo;
    GLTexture2DPtr _scene_color_attach_0;
    GLTexture2DPtr _scene_depth_attach;

    std::shared_ptr<CameraBase> _camera;

    bool _dirty;
    std::string _name;

};

MED_IMAGING_END_NAMESPACE
#endif