#ifndef MED_IMG_SCENE_BASE_H_H
#define MED_IMG_SCENE_BASE_H_H

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgGLResource/mi_gl_resource_define.h"

#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

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

    void download_image_buffer();
    void swap_image_buffer();
    void get_image_buffer(void* buffer);
    

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

    std::unique_ptr<char[]> _image_buffer[2];
    int _front_buffer_id;
    boost::mutex _read_mutex;
    boost::mutex _write_mutex;

};

MED_IMG_END_NAMESPACE
#endif