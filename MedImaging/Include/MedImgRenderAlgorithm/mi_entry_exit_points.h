#ifndef MED_IMAGING_RENDER_ENTRY_EXIT_POINTS_H
#define MED_IMAGING_RENDER_ENTRY_EXIT_POINTS_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"
#include "MedImgGLResource/mi_gl_object.h"
#include "MedImgArithmetic/mi_vector4f.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLTexture2D;
class CameraBase;
class ImageData;
class CameraCalculator;
class RenderAlgo_Export EntryExitPoints
{
public:
    EntryExitPoints();

    virtual ~EntryExitPoints();

    void set_display_size(int iWidth , int iHeight);

    void get_display_size(int& iWidth , int& iHeight);

    void set_strategy( RayCastingStrategy eStrategy );

    virtual void initialize();

    virtual void finialize();

    std::shared_ptr<GLTexture2D> get_entry_points_texture();

    std::shared_ptr<GLTexture2D> get_exit_points_texture();

    Vector4f* get_entry_points_array();

    Vector4f* get_exit_points_array();

    void set_image_data(std::shared_ptr<ImageData> pImgData);

    void set_camera(std::shared_ptr<CameraBase> pCamera);

    void set_camera_calculator(std::shared_ptr<CameraCalculator> pCameraCal);

    virtual void calculate_entry_exit_points() = 0;

public:
    void debug_output_entry_points(const std::string& sFileName);

    void debug_output_exit_points(const std::string& sFileName);

protected:
    std::shared_ptr<GLTexture2D> m_pEntryTex;
    std::shared_ptr<GLTexture2D> m_pExitTex;
    std::unique_ptr<Vector4f[]> m_pEntryBuffer;
    std::unique_ptr<Vector4f[]> m_pExitBuffer;
    int _width;
    int _height;
    std::shared_ptr<CameraBase> m_pCamera;
    std::shared_ptr<ImageData> m_pImgData;
    std::shared_ptr<CameraCalculator> m_pCameraCalculator;
    bool m_bInit;

    RayCastingStrategy m_eStrategy;
};

MED_IMAGING_END_NAMESPACE

#endif