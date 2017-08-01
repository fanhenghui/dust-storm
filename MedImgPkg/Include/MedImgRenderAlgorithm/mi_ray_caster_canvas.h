#ifndef MED_IMG_RAY_CASTER_CANVAS_H_
#define MED_IMG_RAY_CASTER_CANVAS_H_

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgIO/mi_io_define.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export RayCasterCanvas
{
public:
    RayCasterCanvas();

    ~RayCasterCanvas();

    void initialize();

    void set_display_size(int width , int height);

    void get_display_size(int& width, int& height) const;

    GLFBOPtr get_fbo();

    GLTexture2DPtr get_color_attach_texture();

    RGBAUnit* get_color_array();

    //If change data type or display size. Should call it to update FBO and mapped array
    void update_fbo();

    void update_color_array();

public:
    void debug_output_color(const std::string& file_name);

protected:

private:
    GLFBOPtr _gl_fbo;
    GLTexture2DPtr _color_attach_0;//For RGBA Color
    GLTexture2DPtr _depth_attach;
    GLResourceShield _res_shield;

    int _width;
    int _height;
    std::unique_ptr<RGBAUnit[]> _color_array;
    bool _has_init;
};

MED_IMG_END_NAMESPACE


#endif