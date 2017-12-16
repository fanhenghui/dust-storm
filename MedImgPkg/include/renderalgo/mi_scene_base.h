#ifndef MEDIMGRENDERALGO_SCENE_BASE_H_H
#define MEDIMGRENDERALGO_SCENE_BASE_H_H

#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_point2.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_render_algo_export.h"

#include "boost/thread/mutex.hpp"

#ifndef WIN32
#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"
#endif

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export SceneBase {
public:
    SceneBase();
    SceneBase(int width, int height);
    virtual ~SceneBase();

    void set_name(const std::string& name);
    const std::string& get_name() const;

    virtual void initialize();

    virtual void set_display_size(int width, int height);
    void get_display_size(int& width, int& height) const;

    virtual void rotate(const Point2& pre_pt, const Point2& cur_pt);
    virtual void zoom(const Point2& pre_pt, const Point2& cur_pt);
    virtual void pan(const Point2& pre_pt, const Point2& cur_pt);

    std::shared_ptr<CameraBase> get_camera();

    virtual void render();
    void render_to_back();

    void download_image_buffer(bool jpeg = true);
    void swap_image_buffer();
    void get_image_buffer(unsigned char*& buffer, int& size);
    float get_compressing_duration() const;

    void set_dirty(bool flag);
    bool get_dirty() const;

    virtual void set_downsample(bool flag);
    bool get_downsample() const;

    void set_compress_hd_quality(int quality = 80);
    void set_compress_ld_quality(int quality = 15);

protected:
    virtual void pre_render();

protected:
    int _width, _height;

    GLFBOPtr _scene_fbo;
    GLTexture2DPtr _scene_color_attach_0;
    GLTexture2DPtr _scene_color_attach_1;//for fliping verticalily
    GLTexture2DPtr _scene_depth_attach;
    GLResourceShield _res_shield;

    std::shared_ptr<CameraBase> _camera;

    bool _dirty;
    std::string _name;

    std::unique_ptr<unsigned char[]> _image_buffer[2];
    int _image_buffer_size[2];
    int _front_buffer_id;
    boost::mutex _read_mutex;
    boost::mutex _write_mutex;

    bool _downsample;
    int _compress_hd_quality;
    int _compress_ld_quality;

#ifndef WIN32
    // GPU JPEG
    gpujpeg_opengl_texture* _gpujpeg_texture;      // input gpujpeg texture
    gpujpeg_encoder* _gpujpeg_encoder_hd;             // jpeg encoder HD
    gpujpeg_encoder* _gpujpeg_encoder_ld;             // jpeg encoder LD
    gpujpeg_encoder_input _gpujpeg_encoder_input_hd;  // jpeg encoding input HD
    gpujpeg_encoder_input _gpujpeg_encoder_input_ld;  // jpeg encoding input LD
    //在OpenGL的环境下 cuda event
    //会收到OpenGL的影响,导致计算的encoding时间不对(需要在cudaevent之前加一个glfinish)
    float _gpujpeg_encoding_duration;
    // cudaEvent_t _gpujpeg_encoding_start;
    // cudaEvent_t _gpujpeg_encoding_stop;
    bool _gpujpeg_encoder_dirty;

    GLTimeQueryPtr _gl_time_query;
#endif

private:
    DISALLOW_COPY_AND_ASSIGN(SceneBase);
};

MED_IMG_END_NAMESPACE
#endif