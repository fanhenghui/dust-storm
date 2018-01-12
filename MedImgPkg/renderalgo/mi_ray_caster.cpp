#include "mi_ray_caster.h"

#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_texture_1d.h"
#include "glresource/mi_gl_texture_1d_array.h"
#include "glresource/mi_gl_texture_3d.h"
#include "glresource/mi_gl_utils.h"

#include "cudaresource/mi_cuda_utils.h"

#include "mi_ray_caster_canvas.h"
#include "mi_ray_caster_inner_resource.h"
#include "mi_ray_casting_cpu.h"
#include "mi_ray_casting_gpu_gl.h"
#include "mi_ray_casting_gpu_cuda.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

RayCaster::RayCaster(RayCastingStrategy strategy, GPUPlatform gpu_platform)
    : _strategy(strategy), _gpu_platform(gpu_platform), _mask_label_level(L_8),
      _sample_step(0.5f), _custom_sample_step(0.5f), 
      _global_ww(1.0f), _global_wl(0.0f),
      _pseudo_color_array(nullptr), _pseudo_color_length(256), 
      _inner_resource(new RayCasterInnerResource(gpu_platform)),
      _ssd_gray(1.0f), _enable_jittering(false),
      _bound_min(Vector3f(0, 0, 0)), _bound_max(Vector3f(32, 32, 32)),
      _mask_mode(MASK_NONE), 
      _composite_mode(COMPOSITE_AVERAGE),
      _interpolation_mode(LINEAR), 
      _shading_mode(SHADING_NONE),
      _color_inverse_mode(COLOR_INVERSE_DISABLE),
      _mask_overlay_mode(MASK_OVERLAY_DISABLE), 
      _mask_overlay_opacity(0.5f),
      _pre_rendering_duration(0.0f),
      _downsample(false), 
      _expected_fps(30), 
      _map_quarter_canvas(false),
      _test_code(0) {
    _ambient_color[0] = 1.0f;
    _ambient_color[1] = 1.0f;
    _ambient_color[2] = 1.0f;
    _ambient_color[3] = 0.3f;
}

RayCaster::~RayCaster() {}

void RayCaster::render() {
    if (CPU_BASE == _strategy) {
        render_cpu();
    } else if (GPU_BASE == _strategy) {
        if (GL_BASE == _gpu_platform) {
            render_gpu_gl();
        } else {
            render_gpu_cuda();
        }   
    }
}

void RayCaster::on_entry_exit_points_resize(int width, int height) {
    if (GPU_BASE == _strategy && CUDA_BASE == _gpu_platform) {
        if (_ray_casting_gpu_cuda) {
            _ray_casting_gpu_cuda->on_entry_exit_points_resize(width, height);
        }
    }
}

void RayCaster::render_cpu() {
    if (!_ray_casting_cpu) {
        _ray_casting_cpu.reset(new RayCastingCPU(shared_from_this()));
    }
    // clock_t t0 = clock();
    _ray_casting_cpu->render();
    // clock_t t1 = clock();
    // MI_RENDERALGO_LOG(MI_DEBUG) << "Ray casting cost : " << double(t1-t0)/CLOCKS_PER_SEC << "ms.";
}

void RayCaster::render_gpu_gl() {
    if (!_ray_casting_gpu_gl) {
        _ray_casting_gpu_gl.reset(new RayCastingGPUGL(shared_from_this()));
    }

    CHECK_GL_ERROR;
    FBOStack fboStack;
    _canvas->get_fbo()->bind();
    if (this->get_composite_mode() == COMPOSITE_DVR) {
        GLenum buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, buffers);
    }
    else {
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
    }

    // Clear
    glClearColor(0, 0, 0, 0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // downsample strategy
    downsample_adjust();

    // Viewport
    int width, height;
    _canvas->get_display_size(width, height);
    if (_map_quarter_canvas) {
        glViewport(0, 0, width / 2, height / 2);
    }
    else {
        glViewport(0, 0, width, height);
    }

    CHECK_GL_ERROR;

    _ray_casting_gpu_gl->render();

    _pre_rendering_duration = _ray_casting_gpu_gl->get_rendering_duration();
    //MI_RENDERALGO_LOG(MI_DEBUG) << "ray casting cost: " << _pre_rendering_duration << " ms, quarter view: " << _map_quarter_canvas << ", sample step: " << _sample_step << std::endl;
}

void RayCaster::render_gpu_cuda() {
    if (!_ray_casting_gpu_cuda) {
        _ray_casting_gpu_cuda.reset(new RayCastingGPUCUDA(shared_from_this()));
    }
    // downsample strategy
    downsample_adjust();

    CHECK_LAST_CUDA_ERROR;
    _ray_casting_gpu_cuda->render();
    CHECK_LAST_CUDA_ERROR;

    _pre_rendering_duration = _ray_casting_gpu_cuda->get_rendering_duration();
    //MI_RENDERALGO_LOG(MI_DEBUG) << "ray casting cost: " << _pre_rendering_duration << " ms, quarter view: " << _map_quarter_canvas << ", sample step: " << _sample_step << std::endl;    
}

void RayCaster::downsample_adjust(){
    if (!_downsample) {
        _map_quarter_canvas = false;
        _sample_step = _custom_sample_step;
        return;
    }
    
    const static float DURATION_LIMITS = 1.0f; 
    if (_pre_rendering_duration < DURATION_LIMITS) {
        return;
    }
    if (_expected_fps <= 0) {
        return;
    }
    const float exp_duration = 1000.0f / (float)_expected_fps;
    const static float TRIGGER_QUARTER = 1.7f;
    if (_pre_rendering_duration > exp_duration) {
        //downsample
        const float ratio = _pre_rendering_duration / exp_duration;
        if (ratio > TRIGGER_QUARTER && !_map_quarter_canvas) {
            _map_quarter_canvas = true;
        } else if (ratio > TRIGGER_QUARTER && _map_quarter_canvas){
            _sample_step *= TRIGGER_QUARTER;
        } else {
            _sample_step *= ratio;
        }
    } else {
        //upsample
        const float ratio = exp_duration / _pre_rendering_duration;
        if (ratio > TRIGGER_QUARTER && _map_quarter_canvas) {
            _map_quarter_canvas = false;
            _sample_step *= 1.2f;
        } else if (ratio > TRIGGER_QUARTER && !_map_quarter_canvas) {
            _sample_step /= TRIGGER_QUARTER;
        } else {
            _sample_step /= ratio;
        }        
    }
}

void RayCaster::set_volume_data(std::shared_ptr<ImageData> image_data) {
    _volume_data = image_data;
}

void RayCaster::set_mask_data(std::shared_ptr<ImageData> image_data) {
    _mask_data = image_data;
}

void RayCaster::set_volume_data_texture( GPUTexture3DPairPtr volume_textures) {
    _volume_textures = volume_textures;
}

void RayCaster::set_mask_data_texture( GPUTexture3DPairPtr mask_textures) {
    _mask_textures = mask_textures;
}

void RayCaster::set_entry_exit_points(
    std::shared_ptr<EntryExitPoints> entry_exit_points) {
    _entry_exit_points = entry_exit_points;
}

void RayCaster::set_camera(std::shared_ptr<CameraBase> camera) {
    _camera = camera;
}

std::shared_ptr<CameraBase> RayCaster::get_camera() const {
    return _camera;
}

void RayCaster::set_camera_calculator(
    std::shared_ptr<CameraCalculator> camera_cal) {
    _camera_cal = camera_cal;
}

std::shared_ptr<CameraCalculator> RayCaster::get_camera_calculator() const {
    return _camera_cal;
}

void RayCaster::set_sample_step(float sample_step) {
    _sample_step = sample_step;
    _custom_sample_step = sample_step;
}

void RayCaster::set_mask_label_level(LabelLevel label_level) {
    _mask_label_level = label_level;
    _inner_resource->set_mask_label_level(label_level);
}


LabelLevel RayCaster::get_mask_label_level() const {
    return _mask_label_level;
}


void RayCaster::set_visible_labels(std::vector<unsigned char> labels) {
    _inner_resource->set_visible_labels(labels);
}

const std::vector<unsigned char>& RayCaster::get_visible_labels() const {
    return _inner_resource->get_visible_labels();
}

void RayCaster::set_window_level(float ww, float wl, unsigned char label) {
    _inner_resource->set_window_level(ww, wl, label);
}

void RayCaster::set_global_window_level(float ww, float wl) {
    _global_ww = ww;
    _global_wl = wl;
}

void RayCaster::set_pseudo_color_texture(GPUTexture1DPairPtr tex, unsigned int length) {
    _pseudo_color_texture = tex;
    _pseudo_color_length = length;
}

GPUTexture1DPairPtr  RayCaster::get_pseudo_color_texture(unsigned int& length) const {
    length = _pseudo_color_length;
    return _pseudo_color_texture;
}

void RayCaster::set_pseudo_color_array(unsigned char* color_array,
                                       unsigned int length) {
    _pseudo_color_array = color_array;
    _pseudo_color_length = length;
}

void RayCaster::set_color_opacity_texture_array(GPUTexture1DArrayPairPtr tex_array) {
    _color_opacity_texture_array = tex_array;
    _inner_resource->set_color_opacity_texture_array(tex_array);
}

GPUTexture1DArrayPairPtr RayCaster::get_color_opacity_texture_array() const {
    return _color_opacity_texture_array;
}

void RayCaster::set_sillhouette_enhancement() {}

void RayCaster::set_boundary_enhancement() {}

void RayCaster::set_ambient_color(float r, float g, float b, float factor) {
    _ambient_color[0] = r;
    _ambient_color[1] = g;
    _ambient_color[2] = b;
    _ambient_color[3] = factor;
}

void RayCaster::get_ambient_color(float(&rgba)[4]) {
    memcpy(rgba, _ambient_color, 4 * sizeof(float));
}

void RayCaster::set_material(const Material& m, unsigned char label) {
    _inner_resource->set_material(m, label);
}

void RayCaster::set_ssd_gray(float ssd_gray) {
    _ssd_gray = ssd_gray;
}

void RayCaster::set_jittering_enabled(bool flag) {
    _enable_jittering = flag;
}

void RayCaster::set_bounding(const Vector3f& min, const Vector3f& max) {
    _bound_min = min;
    _bound_max = max;
}

void RayCaster::set_clipping_plane_function(
    const std::vector<Vector4f>& funcs) {
    _clipping_planes = funcs;
}

void RayCaster::set_mask_mode(MaskMode mode) {
    _mask_mode = mode;
}

void RayCaster::set_composite_mode(CompositeMode mode) {
    _composite_mode = mode;
}

void RayCaster::set_interpolation_mode(InterpolationMode mode) {
    _interpolation_mode = mode;
}

void RayCaster::set_shading_mode(ShadingMode mode) {
    _shading_mode = mode;
}

void RayCaster::set_color_inverse_mode(ColorInverseMode mode) {
    _color_inverse_mode = mode;
}

void RayCaster::set_canvas(std::shared_ptr<RayCasterCanvas> canvas) {
    _canvas = canvas;
}

std::shared_ptr<RayCasterCanvas> RayCaster::get_canvas() {
    return _canvas;
}

std::shared_ptr<ImageData> RayCaster::get_volume_data() {
    return _volume_data;
}

std::shared_ptr<ImageData> RayCaster::get_mask_data() {
    return _mask_data;
}

GPUTexture3DPairPtr RayCaster::get_volume_data_texture() {
    return _volume_textures;
}

GPUTexture3DPairPtr RayCaster::get_mask_data_texture() {
    return _mask_textures;
}

float RayCaster::get_sample_step() const {
    return _sample_step;
}

void RayCaster::get_global_window_level(float& ww, float& wl) const {
    ww = _global_ww;
    wl = _global_wl;
}

std::shared_ptr<EntryExitPoints> RayCaster::get_entry_exit_points() const {
    return _entry_exit_points;
}

void RayCaster::set_mask_overlay_mode(MaskOverlayMode mode) {
    _mask_overlay_mode = mode;
}

void RayCaster::set_mask_overlay_color(
    std::map<unsigned char, RGBAUnit> colors) {
    _inner_resource->set_mask_overlay_color(colors);
}

void RayCaster::set_mask_overlay_color(RGBAUnit color, unsigned char label) {
    _inner_resource->set_mask_overlay_color(color, label);
}

const std::map<unsigned char, RGBAUnit>&
RayCaster::get_mask_overlay_color() const {
    return _inner_resource->get_mask_overlay_color();
}

std::shared_ptr<RayCasterInnerResource> RayCaster::get_inner_resource() {
    return _inner_resource;
}

MaskMode RayCaster::get_mask_mode() const {
    return _mask_mode;
}

CompositeMode RayCaster::get_composite_mode() const {
    return _composite_mode;
}

InterpolationMode RayCaster::get_interpolation_mode() const {
    return _interpolation_mode;
}

ShadingMode RayCaster::get_shading_mode() const {
    return _shading_mode;
}

ColorInverseMode RayCaster::get_color_inverse_mode() const {
    return _color_inverse_mode;
}

MaskOverlayMode RayCaster::get_mask_overlay_mode() const {
    return _mask_overlay_mode;
}

void RayCaster::set_test_code(int test_code) {
    _test_code = test_code;
}

int RayCaster::get_test_code() const {
    return _test_code;
}

void RayCaster::set_mask_overlay_opacity(float opacity) {
    _mask_overlay_opacity = opacity;
}

float RayCaster::get_mask_overlay_opacity() const {
    return _mask_overlay_opacity;
}

void RayCaster::set_downsample(bool flag) {
    _downsample = flag;
}

bool RayCaster::get_downsample() const {
    return _downsample;
}

void RayCaster::set_expected_fps(int fps) {
    _expected_fps = fps;
}

int RayCaster::get_expected_fps() const {
    return _expected_fps;
}

bool RayCaster::map_quarter_canvas() const {
    return _map_quarter_canvas;
}

MED_IMG_END_NAMESPACE