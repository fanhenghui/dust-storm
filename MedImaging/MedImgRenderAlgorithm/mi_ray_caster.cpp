#include "mi_ray_caster.h"

#include "MedImgGLResource/mi_gl_texture_1d.h"
#include "MedImgGLResource/mi_gl_texture_3d.h"
#include "MedImgGLResource/mi_gl_texture_1d_array.h"
#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster_inner_buffer.h"
#include "mi_ray_casting_cpu.h"
#include "mi_ray_casting_gpu.h"
#include "mi_ray_casting_cpu_brick_acc.h"
#include "mi_ray_caster_canvas.h"

MED_IMAGING_BEGIN_NAMESPACE

RayCaster::RayCaster():_inner_buffer(new RayCasterInnerBuffer()),
_mat_v2w(Matrix4::S_IDENTITY_MATRIX),
_sample_rate(0.5f),
_global_ww(1.0f),
_global_wl(0.0f),
_pseudo_color_array(nullptr),
_pseudo_color_length(256),
_ssd_gray(1.0f),
_enable_jittering(false),
_bound_min(Vector3f(0,0,0)),
_bound_max(Vector3f(32,32,32)),
_mask_mode(MASK_NONE),
_composite_mode(COMPOSITE_AVERAGE),
_interpolation_mode(LINEAR),
_shading_mode(SHADING_NONE),
_color_inverse_mode(COLOR_INVERSE_DISABLE),
_strategy(CPU_BASE),
_brick_corner_array(nullptr),
_volume_brick_unit_array(nullptr),
_mask_brick_unit_array(nullptr),
_volume_brick_info_array(nullptr),
_mask_brick_info_array(nullptr),
_brick_size(32),
_brick_expand(2)
{

}

void RayCaster::initialize()
{
    
}

void RayCaster::finialize()
{
    _inner_buffer->release_buffer();
}

RayCaster::~RayCaster()
{

}

void RayCaster::render(int test_code)
{
    //clock_t t0 = clock();

    if (CPU_BASE == _strategy)
    {
        if (!_ray_casting_cpu)
        {
            _ray_casting_cpu.reset(new RayCastingCPU(shared_from_this()));
        }
        _ray_casting_cpu->render(test_code);
    }
    else if (CPU_BRICK_ACCELERATE == _strategy)
    {
        if (!_ray_casting_cpu_brick_acc)
        {
            _ray_casting_cpu_brick_acc.reset(new RayCastingCPUBrickAcc(shared_from_this()));
        }
        _ray_casting_cpu_brick_acc->render(test_code);
    }
    else if (GPU_BASE == _strategy)
    {
        if (!_ray_casting_gpu)
        {
            _ray_casting_gpu.reset(new RayCastingGPU(shared_from_this()));
        }

        {
            CHECK_GL_ERROR;

            FBOStack fboStack;
            _canvas->get_fbo()->bind();

            CHECK_GL_ERROR;

            glDrawBuffer(GL_COLOR_ATTACHMENT0);

            CHECK_GL_ERROR;

            //Clear 
            glClearColor(0,0,0,0);
            glClearDepth(1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            //Viewport
            int width , height;
            _canvas->get_display_size(width , height);
            glViewport(0,0,width , height);

            CHECK_GL_ERROR;

            _ray_casting_gpu->render(test_code);
        }
    }

    //clock_t t1 = clock();
    //std::cout << "<<<>>><<<>>><<<>>><<<>>><<<>>>\n";
    //std::cout << "Ray casting cost : " << double(t1-t0) << "ms\n";
    //std::cout << "<<<>>><<<>>><<<>>><<<>>><<<>>>\n";
}

void RayCaster::set_volume_data(std::shared_ptr<ImageData> image_data)
{
    _volume_data = image_data;
}

void RayCaster::set_mask_data(std::shared_ptr<ImageData> image_data)
{
    _mask_data = image_data;
}

void RayCaster::set_volume_data_texture(std::vector<GLTexture3DPtr> volume_textures)
{
    _volume_textures = volume_textures;
}

void RayCaster::set_mask_data_texture(std::vector<GLTexture3DPtr> mask_textures)
{
    _mask_textures =mask_textures;
}

void RayCaster::set_entry_exit_points(std::shared_ptr<EntryExitPoints> entry_exit_points)
{
    _entry_exit_points = entry_exit_points;
}

void RayCaster::set_camera(std::shared_ptr<CameraBase> camera)
{
    _camera = camera;
}

void RayCaster::set_volume_to_world_matrix(const Matrix4& mat)
{
    _mat_v2w = mat;
}

void RayCaster::set_sample_rate(float sample_rate)
{
    _sample_rate = sample_rate;
}

void RayCaster::set_mask_label_level(LabelLevel label_level)
{
    _inner_buffer->set_mask_label_level(label_level);
}

void RayCaster::set_visible_labels(std::vector<unsigned char> labels)
{
    _inner_buffer->set_visible_labels(labels);
}

void RayCaster::set_window_level(float ww , float wl , unsigned char label)
{
    _inner_buffer->set_window_level(ww , wl , label);
}

void RayCaster::set_global_window_level(float ww , float wl)
{
    _global_ww = ww;
    _global_wl = wl;
}

void RayCaster::set_pseudo_color_texture(GLTexture1DPtr tex , unsigned int length)
{
    _pseudo_color_texture = tex;
    _pseudo_color_length= length;
}

GLTexture1DPtr RayCaster::get_pseudo_color_texture(unsigned int& length) const
{
    length = _pseudo_color_length;
    return _pseudo_color_texture;
}

void RayCaster::set_pseudo_color_array(unsigned char* color_array , unsigned int length)
{
    _pseudo_color_array = color_array;
    _pseudo_color_length = length;
}

void RayCaster::set_transfer_function_texture(GLTexture1DArrayPtr tex_array)
{
    _transfer_function = tex_array;
}

void RayCaster::set_sillhouette_enhancement()
{

}

void RayCaster::set_boundary_enhancement()
{

}

void RayCaster::set_material()
{

}

void RayCaster::set_light_color()
{

}

void RayCaster::set_light_factor()
{

}

void RayCaster::set_ssd_gray(float ssd_gray)
{
    _ssd_gray = ssd_gray;
}

void RayCaster::set_jittering_enabled(bool flag)
{
    _enable_jittering = flag;
}

void RayCaster::set_bounding(const Vector3f& min, const Vector3f& max)
{
    _bound_min = min;
    _bound_max = max;
}

void RayCaster::set_clipping_plane_function(const std::vector<Vector4f> &funcs)
{
    _clipping_planes = funcs;
}

void RayCaster::set_mask_mode(MaskMode mode)
{
    _mask_mode = mode;
}

void RayCaster::set_composite_mode(CompositeMode mode)
{
    _composite_mode = mode;
}

void RayCaster::set_interpolation_mode(InterpolationMode mode)
{
    _interpolation_mode = mode;
}

void RayCaster::set_shading_mode(ShadingMode mode)
{
    _shading_mode = mode;
}

void RayCaster::set_color_inverse_mode(ColorInverseMode mode)
{
    _color_inverse_mode = mode;
}

void RayCaster::set_canvas(std::shared_ptr<RayCasterCanvas> canvas)
{
    _canvas = canvas;
}

void RayCaster::set_strategy(RayCastingStrategy strategy)
{
    _strategy = strategy;
}

void RayCaster::set_brick_size(unsigned int brick_size)
{
    _brick_size = brick_size;
}

void RayCaster::set_brick_expand(unsigned int brick_expand)
{
    _brick_expand = brick_expand;
}

void RayCaster::set_brick_corner(BrickCorner* brick_corner_array)
{
    _brick_corner_array = brick_corner_array;
}

void RayCaster::set_volume_brick_unit(BrickUnit* volume_brick_unit_array)
{
    _volume_brick_unit_array = volume_brick_unit_array;
}

void RayCaster::set_mask_brick_unit(BrickUnit* mask_brick_unit_array)
{
    _mask_brick_unit_array = mask_brick_unit_array;
}

void RayCaster::set_mask_brick_info(MaskBrickInfo* mask_brick_info_array)
{
    _mask_brick_info_array = mask_brick_info_array;
}

void RayCaster::set_volume_brick_info(VolumeBrickInfo* volume_brick_info_array)
{
    _volume_brick_info_array = volume_brick_info_array;
}

const std::vector<BrickDistance>& RayCaster::get_brick_distance() const
{
    return _ray_casting_cpu_brick_acc->get_brick_distance();
}

unsigned int RayCaster::get_ray_casting_brick_count() const
{
    return _ray_casting_cpu_brick_acc->get_ray_casting_brick_count();
}

std::shared_ptr<ImageData> RayCaster::get_volume_data()
{
    return _volume_data;
}

std::vector<GLTexture3DPtr> RayCaster::get_volume_data_texture()
{
    return _volume_textures;
}

float RayCaster::get_sample_rate() const
{
    return _sample_rate;
}

void RayCaster::get_global_window_level(float& ww , float& wl) const
{
    ww = _global_ww;
    wl = _global_wl;
}

std::shared_ptr<EntryExitPoints> RayCaster::get_entry_exit_points() const
{
    return _entry_exit_points;
}



MED_IMAGING_END_NAMESPACE