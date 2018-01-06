#ifndef MEDIMGRENDERALGO_SCENE_BASE_H_H
#define MEDIMGRENDERALGO_SCENE_BASE_H_H

#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_point2.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_render_algo_export.h"
#include "renderalgo/mi_ray_caster_define.h"

#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE

class GPUImgCompressor;
class RenderAlgo_Export SceneBase {
public:
    SceneBase(GPUPlatform platfrom);
    SceneBase(GPUPlatform platfrom, int width, int height);
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
    virtual void render_to_back();

    void download_image_buffer(bool jpeg = true);
    void swap_image_buffer();
    void get_image_buffer(unsigned char*& buffer, int& size);

    void set_dirty(bool flag);
    bool get_dirty() const;

    virtual void set_downsample(bool flag);
    bool get_downsample() const;

    void set_compress_hd_quality(int quality = 80);
    void set_compress_ld_quality(int quality = 15);
    float get_compressing_duration() const;

protected:
    virtual void pre_render();

protected:
    int _width, _height;
    GPUPlatform _gpu_platform;

    GLFBOPtr _scene_fbo;
    GLTexture2DPtr _scene_color_attach_0;//for rendering
    GLTexture2DPtr _scene_color_attach_1;//for flip vertical color-0 to compress out
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

    //compressor
    int _compress_hd_quality;
    int _compress_ld_quality;
    std::shared_ptr<GPUImgCompressor> _compressor;
    bool _gpujpeg_encoder_dirty;

private:
    DISALLOW_COPY_AND_ASSIGN(SceneBase);
};

MED_IMG_END_NAMESPACE
#endif