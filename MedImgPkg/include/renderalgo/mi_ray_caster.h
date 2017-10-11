#ifndef MEDIMGRENDERALGO_RAY_CASTER_H
#define MEDIMGRENDERALGO_RAY_CASTER_H

#include "renderalgo/mi_brick_define.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_render_algo_export.h"

#include "arithmetic/mi_matrix4.h"
#include "arithmetic/mi_vector4f.h"

#include "glresource/mi_gl_resource_define.h"

MED_IMG_BEGIN_NAMESPACE

class EntryExitPoints;
class ImageData;
class CameraBase;
class CameraCalculator;
class RayCasterInnerBuffer;
class RayCasterCanvas;
class RayCastingCPU;
class RayCastingGPU;
class RayCastingCPUBrickAcc;

class RenderAlgo_Export RayCaster
        : public std::enable_shared_from_this<RayCaster> {
    friend class RayCastingCPU;
    friend class RayCastingCPUBrickAcc;
    friend class RayCastingGPU;

public:
    RayCaster();

    ~RayCaster();

    void render();

    void set_test_code(int test_code);
    int get_test_code() const;

    // Ray casting strategy
    void set_strategy(RayCastingStrategy strategy);

    void set_canvas(std::shared_ptr<RayCasterCanvas> canvas);

    // Mask label level
    // Default is L_8
    void set_mask_label_level(LabelLevel label_level);

    //////////////////////////////////////////////////////////////////////////
    // Input data
    //////////////////////////////////////////////////////////////////////////

    // Volume & mask texture/array
    void set_volume_data(std::shared_ptr<ImageData> image_data);
    std::shared_ptr<ImageData> get_volume_data();

    void set_mask_data(std::shared_ptr<ImageData> image_data);
    std::shared_ptr<ImageData> get_mask_data();

    void set_volume_data_texture(std::vector<GLTexture3DPtr> volume_textures);
    std::vector<GLTexture3DPtr> get_volume_data_texture();

    void set_mask_data_texture(std::vector<GLTexture3DPtr> mask_textures);
    std::vector<GLTexture3DPtr> get_mask_data_texture();

    // Entry exit points
    void
    set_entry_exit_points(std::shared_ptr<EntryExitPoints> entry_exit_points);

    std::shared_ptr<EntryExitPoints> get_entry_exit_points() const;

    //////////////////////////////////////////////////////////////////////////
    // Ray casting parameter
    //////////////////////////////////////////////////////////////////////////

    // Volume modeling parameter
    void set_camera(std::shared_ptr<CameraBase> camera);
    std::shared_ptr<CameraBase> get_camera() const;

    void set_camera_calculator(std::shared_ptr<CameraCalculator> camera_cal);
    std::shared_ptr<CameraCalculator> get_camera_calculator() const;

    // Sample rate
    void set_sample_rate(float sample_rate);
    float get_sample_rate() const;

    // Label parameter
    void set_visible_labels(std::vector<unsigned char> labels);
    const std::vector<unsigned char>& get_visible_labels() const;

    // Window level parameter
    // Here
    void set_window_level(float ww, float wl, unsigned char label);
    void set_global_window_level(float ww, float wl);

    void get_global_window_level(float& ww, float& wl) const;

    // Pseudo color parameter
    void set_pseudo_color_texture(GLTexture1DPtr tex, unsigned int length);
    GLTexture1DPtr get_pseudo_color_texture(unsigned int& length) const;
    void set_pseudo_color_array(unsigned char* color_array, unsigned int length);

    void set_color_opacity_texture_array(GLTexture1DArrayPtr tex_array);
    GLTexture1DArrayPtr get_color_opacity_texture_array() const;

    // Mask overlay color
    void set_mask_overlay_color(std::map<unsigned char, RGBAUnit> colors);
    void set_mask_overlay_color(RGBAUnit color, unsigned char label);
    const std::map<unsigned char, RGBAUnit>& get_mask_overlay_color() const;
    void set_mask_overlay_opacity(float opacity);
    float get_mask_overlay_opacity() const;

    // Enhancement parameter
    void set_sillhouette_enhancement();
    void set_boundary_enhancement();

    // Shading parameter
    void set_ambient_color(float r, float g, float b, float factor);
    void get_ambient_color(float(&rgba)[4]);
    void set_material(const Material& m, unsigned char label);

    // SSD gray value
    void set_ssd_gray(float ssd_gray);

    // Jittering to prevent wooden artifacts
    void set_jittering_enabled(bool flag);

    // Bounding box
    void set_bounding(const Vector3f& min, const Vector3f& max);

    // Clipping plane
    void set_clipping_plane_function(const std::vector<Vector4f>& funcs);

    // Ray casting mode parameter
    void set_mask_mode(MaskMode mode);
    void set_composite_mode(CompositeMode mode);
    void set_interpolation_mode(InterpolationMode mode);
    void set_shading_mode(ShadingMode mode);
    void set_color_inverse_mode(ColorInverseMode mode);
    void set_mask_overlay_mode(MaskOverlayMode mode);

    MaskMode get_mask_mode() const;
    CompositeMode get_composite_mode() const;
    InterpolationMode get_interpolation_mode() const;
    ShadingMode get_shading_mode() const;
    ColorInverseMode get_color_inverse_mode() const;
    MaskOverlayMode get_mask_overlay_mode() const;

    // Inner buffer
    std::shared_ptr<RayCasterInnerBuffer> get_inner_buffer();

protected:
    // Input data
    std::shared_ptr<ImageData> _volume_data;
    std::vector<GLTexture3DPtr> _volume_textures;

    std::shared_ptr<ImageData> _mask_data;
    std::vector<GLTexture3DPtr> _mask_textures;

    // Entry exit points
    std::shared_ptr<EntryExitPoints> _entry_exit_points;

    std::shared_ptr<CameraBase> _camera;
    std::shared_ptr<CameraCalculator> _camera_cal;

    // Data sample rate(DVR 0.5 , MIPs 1.0)
    float _sample_rate;

    // Global window level for MIPs mode
    float _global_ww;
    float _global_wl;

    // Transfer function & pseudo color
    GLTexture1DArrayPtr _color_opacity_texture_array;
    GLTexture1DPtr _pseudo_color_texture;
    unsigned char* _pseudo_color_array;
    unsigned int _pseudo_color_length;

    // Inner buffer to contain label based parameter
    std::shared_ptr<RayCasterInnerBuffer> _inner_buffer;

    // SSD gray value
    float _ssd_gray;

    // DVR jittering flag
    bool _enable_jittering;

    // Bounding
    Vector3f _bound_min;
    Vector3f _bound_max;

    // Clipping plane
    std::vector<Vector4f> _clipping_planes;

    // Ray casting mode
    MaskMode _mask_mode;
    CompositeMode _composite_mode;
    InterpolationMode _interpolation_mode;
    ShadingMode _shading_mode;
    ColorInverseMode _color_inverse_mode;
    MaskOverlayMode _mask_overlay_mode;

    // Mask overlay opacity
    float _mask_overlay_opacity;

    // Ambient
    float _ambient_color[4];

    // Processing unit type
    RayCastingStrategy _strategy;

    std::shared_ptr<RayCastingCPU> _ray_casting_cpu;
    std::shared_ptr<RayCastingGPU> _ray_casting_gpu;

    // Canvas for rendering
    std::shared_ptr<RayCasterCanvas> _canvas;

    // Test code for debug
    int _test_code;
};

MED_IMG_END_NAMESPACE

#endif