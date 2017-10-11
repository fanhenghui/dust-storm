#include "mi_ray_caster.h"

#include "glresource/mi_gl_fbo.h"
#include "glresource/mi_gl_texture_1d.h"
#include "glresource/mi_gl_texture_1d_array.h"
#include "glresource/mi_gl_texture_3d.h"
#include "glresource/mi_gl_utils.h"

#include "mi_ray_caster_canvas.h"
#include "mi_ray_caster_inner_buffer.h"
#include "mi_ray_casting_cpu.h"
#include "mi_ray_casting_gpu.h"

MED_IMG_BEGIN_NAMESPACE

RayCaster::RayCaster()
    : _inner_buffer(new RayCasterInnerBuffer()), _sample_rate(0.5f),
      _global_ww(1.0f), _global_wl(0.0f), _pseudo_color_array(nullptr),
      _pseudo_color_length(256), _ssd_gray(1.0f), _enable_jittering(false),
      _bound_min(Vector3f(0, 0, 0)), _bound_max(Vector3f(32, 32, 32)),
      _mask_mode(MASK_NONE), _composite_mode(COMPOSITE_AVERAGE),
      _interpolation_mode(LINEAR), _shading_mode(SHADING_NONE),
      _color_inverse_mode(COLOR_INVERSE_DISABLE),
      _mask_overlay_mode(MASK_OVERLAY_DISABLE), _strategy(CPU_BASE),
      _test_code(0),
      _mask_overlay_opacity(0.5f) {
    _ambient_color[0] = 1.0f;
    _ambient_color[1] = 1.0f;
    _ambient_color[2] = 1.0f;
    _ambient_color[3] = 0.3f;
}

RayCaster::~RayCaster() {}

void RayCaster::render() {
    // clock_t t0 = clock();
    if (CPU_BASE == _strategy) {
        if (!_ray_casting_cpu) {
            _ray_casting_cpu.reset(new RayCastingCPU(shared_from_this()));
        }

        _ray_casting_cpu->render();
    } else if (GPU_BASE == _strategy) {
        if (!_ray_casting_gpu) {
            _ray_casting_gpu.reset(new RayCastingGPU(shared_from_this()));
        }

        {
            CHECK_GL_ERROR;

            FBOStack fboStack;
            _canvas->get_fbo()->bind();
            glDrawBuffer(GL_COLOR_ATTACHMENT0);

            // Clear
            glClearColor(0, 0, 0, 0);
            glClearDepth(1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Viewport
            int width, height;
            _canvas->get_display_size(width, height);
            glViewport(0, 0, width, height);

            CHECK_GL_ERROR;

            _ray_casting_gpu->render();
        }
    }
    // clock_t t1 = clock();
    // MI_RENDERALGO_LOG(MI_DEBUG) << "Ray casting cost : " << double(t1-t0) << "ms.";
}

void RayCaster::set_volume_data(std::shared_ptr<ImageData> image_data) {
    _volume_data = image_data;
}

void RayCaster::set_mask_data(std::shared_ptr<ImageData> image_data) {
    _mask_data = image_data;
}

void RayCaster::set_volume_data_texture(
    std::vector<GLTexture3DPtr> volume_textures) {
    _volume_textures = volume_textures;
}

void RayCaster::set_mask_data_texture(
    std::vector<GLTexture3DPtr> mask_textures) {
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

void RayCaster::set_sample_rate(float sample_rate) {
    _sample_rate = sample_rate;
}

void RayCaster::set_mask_label_level(LabelLevel label_level) {
    _inner_buffer->set_mask_label_level(label_level);
}

void RayCaster::set_visible_labels(std::vector<unsigned char> labels) {
    _inner_buffer->set_visible_labels(labels);
}

const std::vector<unsigned char>& RayCaster::get_visible_labels() const {
    return _inner_buffer->get_visible_labels();
}

void RayCaster::set_window_level(float ww, float wl, unsigned char label) {
    _inner_buffer->set_window_level(ww, wl, label);
}

void RayCaster::set_global_window_level(float ww, float wl) {
    _global_ww = ww;
    _global_wl = wl;
}

void RayCaster::set_pseudo_color_texture(GLTexture1DPtr tex,
        unsigned int length) {
    _pseudo_color_texture = tex;
    _pseudo_color_length = length;
}

GLTexture1DPtr RayCaster::get_pseudo_color_texture(unsigned int& length) const {
    length = _pseudo_color_length;
    return _pseudo_color_texture;
}

void RayCaster::set_pseudo_color_array(unsigned char* color_array,
                                       unsigned int length) {
    _pseudo_color_array = color_array;
    _pseudo_color_length = length;
}

void RayCaster::set_color_opacity_texture_array(GLTexture1DArrayPtr tex_array) {
    _color_opacity_texture_array = tex_array;
}

GLTexture1DArrayPtr RayCaster::get_color_opacity_texture_array() const {
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
    _inner_buffer->set_material(m, label);
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

void RayCaster::set_strategy(RayCastingStrategy strategy) {
    _strategy = strategy;
}

std::shared_ptr<ImageData> RayCaster::get_volume_data() {
    return _volume_data;
}

std::shared_ptr<ImageData> RayCaster::get_mask_data() {
    return _mask_data;
}

std::vector<GLTexture3DPtr> RayCaster::get_volume_data_texture() {
    return _volume_textures;
}

float RayCaster::get_sample_rate() const {
    return _sample_rate;
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
    _inner_buffer->set_mask_overlay_color(colors);
}

void RayCaster::set_mask_overlay_color(RGBAUnit color, unsigned char label) {
    _inner_buffer->set_mask_overlay_color(color, label);
}

const std::map<unsigned char, RGBAUnit>&
RayCaster::get_mask_overlay_color() const {
    return _inner_buffer->get_mask_overlay_color();
}

std::shared_ptr<RayCasterInnerBuffer> RayCaster::get_inner_buffer() {
    return _inner_buffer;
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

std::vector<GLTexture3DPtr> RayCaster::get_mask_data_texture() {
    return _mask_textures;
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

MED_IMG_END_NAMESPACE