#ifndef MEDIMGRENDERALGO_ENTRY_EXIT_POINTS_H
#define MEDIMGRENDERALGO_ENTRY_EXIT_POINTS_H

#include "renderalgo/mi_render_algo_export.h"

#include "arithmetic/mi_vector4f.h"
#include "glresource/mi_gl_resource_define.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "renderalgo/mi_ray_caster_define.h"
#include "renderalgo/mi_gpu_resource_pair.h"

MED_IMG_BEGIN_NAMESPACE

class CameraBase;
class ImageData;
class CameraCalculator;
class RenderAlgo_Export EntryExitPoints
        : public std::enable_shared_from_this<EntryExitPoints> {
public:
    EntryExitPoints(RayCastingStrategy s, GPUPlatform p);
    virtual ~EntryExitPoints();

    GPUPlatform get_gpu_platform() const;

    virtual void set_display_size(int width, int height);
    void get_display_size(int& width, int& height);

    virtual void initialize();

    GPUCanvasPairPtr get_entry_points_texture();
    GPUCanvasPairPtr get_exit_points_texture();

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
    RayCastingStrategy _strategy;
    GPUPlatform _gpu_platform;

    GPUCanvasPairPtr _entry_points_texture;
    GPUCanvasPairPtr _exit_points_texture;
    GLResourceShield _res_shield;

    std::unique_ptr<Vector4f[]> _entry_points_buffer;
    std::unique_ptr<Vector4f[]> _exit_points_buffer;
    int _width;
    int _height;

    std::shared_ptr<CameraBase> _camera;
    std::shared_ptr<ImageData> _volume_data;
    std::shared_ptr<CameraCalculator> _camera_calculator;

    bool _has_init;    
};

MED_IMG_END_NAMESPACE

#endif