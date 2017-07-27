#ifndef MED_IMG_RENDER_ENTRY_EXIT_POINTS_H
#define MED_IMG_RENDER_ENTRY_EXIT_POINTS_H

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgArithmetic/mi_vector4f.h"

MED_IMG_BEGIN_NAMESPACE

class CameraBase;
class ImageData;
class CameraCalculator;
class RenderAlgo_Export EntryExitPoints
{
public:
    EntryExitPoints();
    virtual ~EntryExitPoints();

    void set_display_size(int width , int height);
    void get_display_size(int& width , int& height);

    void set_strategy( RayCastingStrategy strategy );

    virtual void initialize();
    virtual void finialize();

    GLTexture2DPtr get_entry_points_texture();
    GLTexture2DPtr get_exit_points_texture();

    Vector4f* get_entry_points_array();
    Vector4f* get_exit_points_array();

    void set_volume_data(std::shared_ptr<ImageData> image_data);
    std::shared_ptr<ImageData> get_volume_data() const;

    void set_camera(std::shared_ptr<CameraBase> camera);
    std::shared_ptr<CameraBase> get_camera() const;

    void set_camera_calculator(std::shared_ptr<CameraCalculator> camera_cal);
    std::shared_ptr<CameraCalculator> get_camera_calculator() const;

    virtual void calculate_entry_exit_points() = 0;

public:
    void debug_output_entry_points(const std::string& file_name);

    void debug_output_exit_points(const std::string& file_name);

protected:
    GLTexture2DPtr _entry_points_texture;
    GLTexture2DPtr _exit_points_texture;
    std::unique_ptr<Vector4f[]> _entry_points_buffer;
    std::unique_ptr<Vector4f[]> _exit_points_buffer;
    int _width;
    int _height;
    std::shared_ptr<CameraBase> _camera;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<CameraCalculator> _camera_calculator;
    bool _has_init;

    RayCastingStrategy _strategy;
};

MED_IMG_END_NAMESPACE

#endif