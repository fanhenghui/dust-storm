#ifndef MED_IMAGING_RAY_CASTER_H_
#define MED_IMAGING_RAY_CASTER_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_vector4f.h"

#include "MedImgGLResource/mi_gl_resource_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class EntryExitPoints;
class ImageData;
class CameraBase;
class RayCasterInnerBuffer;
class RayCasterCanvas;

class RenderAlgo_Export RayCaster : public std::enable_shared_from_this<RayCaster>
{
    friend class RayCastingCPU;
    friend class RayCastingCPUBrickAcc;
    friend class RayCastingGPU;

public:
    RayCaster();

    ~RayCaster();

    void initialize();

    void finialize();

    void render(int test_code = 0);

    //Ray casting strategy
    void set_strategy(RayCastingStrategy strategy);

    void set_canvas(std::shared_ptr<RayCasterCanvas> canvas);

    //Mask label level
    //Default is L_8
    void set_mask_label_level(LabelLevel label_level);

    //////////////////////////////////////////////////////////////////////////
    //Input data
    //////////////////////////////////////////////////////////////////////////

    //Volume & mask texture/array
    void set_volume_data(std::shared_ptr<ImageData> image_data);
    void set_mask_data(std::shared_ptr<ImageData> image_data);
    void set_volume_data_texture(std::vector<GLTexture3DPtr> volume_textures);
    void set_mask_data_texture(std::vector<GLTexture3DPtr> mask_textures);

    std::shared_ptr<ImageData> get_volume_data();
    std::vector<GLTexture3DPtr> get_volume_data_texture( );

    //Brick acceleration
    void set_brick_size(unsigned int brick_size);
    void set_brick_expand(unsigned int brick_expand);
    void set_brick_corner(BrickCorner* brick_corner_array);
    void set_volume_brick_unit(BrickUnit* volume_brick_unit_array);
    void set_mask_brick_unit(BrickUnit* mask_brick_unit_array);
    void set_volume_brick_info(VolumeBrickInfo* volume_brick_info_array);
    void set_mask_brick_info(MaskBrickInfo* mask_brick_info_array);//Here don't need label visibility status , just set current info

    //Entry exit points
    void set_entry_exit_points(std::shared_ptr<EntryExitPoints> entry_exit_points);

    std::shared_ptr<EntryExitPoints> get_entry_exit_points() const;


    //////////////////////////////////////////////////////////////////////////
    //Ray casting parameter
    //////////////////////////////////////////////////////////////////////////

    //Volume modeling parameter
    void set_camera(std::shared_ptr<CameraBase> camera);
    void set_volume_to_world_matrix(const Matrix4& mat);

    //Sample rate
    void set_sample_rate(float sample_rate);

    float get_sample_rate() const;

    //Label parameter
    void set_visible_labels(std::vector<unsigned char> labels);
    const std::vector<unsigned char>& get_visible_labels() const; 

    //Window level parameter
    //Here
    void set_window_level(float ww , float wl , unsigned char label);
    void set_global_window_level(float ww , float wl);

    void get_global_window_level(float& ww , float& wl) const;

    //Transfer function & pseudo color parameter
    void set_pseudo_color_texture(GLTexture1DPtr tex , unsigned int length);
    GLTexture1DPtr get_pseudo_color_texture(unsigned int& length) const;

    //RGB8 array
    void set_pseudo_color_array(unsigned char* color_array , unsigned int length);
    void set_transfer_function_texture(GLTexture1DArrayPtr tex_array);

    //Mask overlay color
    void set_mask_overlay_color(std::map<unsigned char , RGBAUnit> colors);
    void set_mask_overlay_color(RGBAUnit color , unsigned char label);
    const std::map<unsigned char , RGBAUnit>& get_mask_overlay_color() const;

    //Enhancement parameter
    void set_sillhouette_enhancement();
    void set_boundary_enhancement();

    //Shading parameter
    void set_material();
    void set_light_color();
    void set_light_factor();

    //SSD gray value
    void set_ssd_gray(float ssd_gray);

    //Jittering to prevent wooden artifacts
    void set_jittering_enabled(bool flag);

    //Bounding box
    void set_bounding(const Vector3f& min, const Vector3f& max);

    //Clipping plane
    void set_clipping_plane_function(const std::vector<Vector4f> &funcs);

    //Ray casting mode parameter
    void set_mask_mode(MaskMode mode);
    void set_composite_mode(CompositeMode mode);
    void set_interpolation_mode(InterpolationMode mode);
    void set_shading_mode(ShadingMode mode);
    void set_color_inverse_mode(ColorInverseMode mode);
    void set_mask_overlay_mode(MaskOverlayMode mode);

    //Inner buffer
    std::shared_ptr<RayCasterInnerBuffer> get_inner_buffer();


    //////////////////////////////////////////////////////////////////////////
    //For testing
    const std::vector<BrickDistance>& get_brick_distance() const;
    unsigned int get_ray_casting_brick_count() const;

protected:
    //Input data
    std::shared_ptr<ImageData> _volume_data;
    std::vector<GLTexture3DPtr> _volume_textures;

    std::shared_ptr<ImageData> _mask_data;
    std::vector<GLTexture3DPtr> _mask_textures;

    //Brick acceleration
    BrickCorner* _brick_corner_array;
    BrickUnit* _volume_brick_unit_array;
    BrickUnit* _mask_brick_unit_array;
    VolumeBrickInfo* _volume_brick_info_array;
    MaskBrickInfo* _mask_brick_info_array;
    unsigned int _brick_size;
    unsigned int _brick_expand;

    //Entry exit points
    std::shared_ptr<EntryExitPoints> _entry_exit_points;

    std::shared_ptr<CameraBase> _camera;

    Matrix4 _mat_v2w;

    //Data sample rate(DVR 0.5 , MIPs 1.0) 
    float _sample_rate;

    //Global window level for MIPs mode
    float _global_ww;
    float _global_wl;

    //Transfer function & pseudo color 
    GLTexture1DArrayPtr _transfer_function;
    GLTexture1DPtr _pseudo_color_texture;
    unsigned char* _pseudo_color_array;
    unsigned int _pseudo_color_length;

    //Inner buffer to contain label based parameter
    std::shared_ptr<RayCasterInnerBuffer> _inner_buffer;

    //SSD gray value
    float _ssd_gray;

    //DVR jittering flag
    bool _enable_jittering;

    //Bounding
    Vector3f _bound_min;
    Vector3f _bound_max;

    //Clipping plane
    std::vector<Vector4f> _clipping_planes;

    //Ray casting mode
    MaskMode _mask_mode;
    CompositeMode _composite_mode;
    InterpolationMode _interpolation_mode;
    ShadingMode _shading_mode;
    ColorInverseMode _color_inverse_mode;
    MaskOverlayMode _mask_overlay_mode;

    //Processing unit type
    RayCastingStrategy _strategy;

    std::shared_ptr<RayCastingCPU> _ray_casting_cpu;
    std::shared_ptr<RayCastingGPU> _ray_casting_gpu;
    std::shared_ptr<RayCastingCPUBrickAcc> _ray_casting_cpu_brick_acc;

    //Canvas for rendering
    std::shared_ptr<RayCasterCanvas> _canvas;
};

MED_IMAGING_END_NAMESPACE


#endif