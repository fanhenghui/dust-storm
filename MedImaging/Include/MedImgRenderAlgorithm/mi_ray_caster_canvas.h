#ifndef MED_IMAGING_RAY_CASTER_CANVAS_H_
#define MED_IMAGING_RAY_CASTER_CANVAS_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgCommon/mi_common_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export RayCasterCanvas
{
public:
    RayCasterCanvas();

    ~RayCasterCanvas();

    void initialize();

    void finialize();

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
    int _width;
    int _height;
    std::unique_ptr<RGBAUnit[]> _color_array;
    bool _has_init;
};

MED_IMAGING_END_NAMESPACE


#endif