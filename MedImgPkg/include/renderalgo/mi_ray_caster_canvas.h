#ifndef MEDIMGRENDERALGO_RAY_CASTER_CANVAS_H
#define MEDIMGRENDERALGO_RAY_CASTER_CANVAS_H

#include "renderalgo/mi_render_algo_export.h"

#include "arithmetic/mi_color_unit.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_gpu_resource_pair.h"
#include "renderalgo/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export RayCasterCanvas {
public:
    RayCasterCanvas(RayCastingStrategy strategy, GPUPlatform p);

    ~RayCasterCanvas();

    //Multi-color-attach just for VR
    void initialize(bool multi_color_attach = false);

    void set_display_size(int width, int height);

    void get_display_size(int& width, int& height) const;

    GLFBOPtr get_fbo();

    GPUCanvasPairPtr get_color_attach_texture(int id=0);

    RGBAUnit* get_color_array();

    void update_color_array();

public:
    void debug_output_color_0(const std::string& file_name);
    void debug_output_color_1(const std::string& file_name);

protected:
private:
    RayCastingStrategy _stratrgy;
    GPUPlatform _gpu_platform;

    GLFBOPtr          _gl_fbo;
    GLTexture2DPtr    _gl_depth_attach;
    GLResourceShield  _res_shield;

    GPUCanvasPairPtr _color_attach_0;// For RGBA Color
    GPUCanvasPairPtr _color_attach_1;// For VR post ray casting(EG: save ray stop position)

    bool _has_init;

    int _width;
    int _height;
    std::unique_ptr<RGBAUnit[]> _color_array;//FOR CPU MPR
};

MED_IMG_END_NAMESPACE

#endif