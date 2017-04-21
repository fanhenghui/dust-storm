#ifndef MED_IMAGING_ENTRY_EXIT_POINTS_H_
#define MED_IMAGING_ENTRY_EXIT_POINTS_H_

#include "MedImgRenderAlgorithm/mi_entry_exit_points.h"
#include "MedImgCommon/mi_common_define.h"
#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_vector4f.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export MPREntryExitPoints : public EntryExitPoints
{
public:
    MPREntryExitPoints();

    virtual ~MPREntryExitPoints();

    virtual void initialize();

    virtual void finialize();

    void set_thickness(float thickness);// In volume coordinate

    void set_sample_rate(float sample_rate);

    virtual void calculate_entry_exit_points();

    //////////////////////////////////////////////////////////////////////////
    /// \ plane function :  ax + by + cz = d
    /// \ Or normal&point x*N = d , thus d is the distance between original(0,0,0) and the plane
    /// \ vector4f (a,b,c,d)
    ///\ Entry plane's normal is form entry to exit
    ///\ Exit plane's normal is form exit to entry
    /// \In volume coordinate
    //////////////////////////////////////////////////////////////////////////
    void get_entry_exit_plane(Vector4f& vEntry , Vector4f& vExit , Vector3f& vRayDirNorm);

private:
    void cal_entry_exit_points_cpu_i();

    void cal_entry_exit_plane_cpu_i();

    void cal_entry_exit_points_gpu_i();

private:
    
    float _thickness;
    float _sample_rate;

    //Entry exit plane(orthogonal)
    Vector4f _entry_plane;
    Vector4f _exit_plane;
    Vector3f _ray_dir_norm;

    float _standard_steps;

    //GPU entry exit points cal
    GLProgramPtr _program;

};


MED_IMAGING_END_NAMESPACE

#endif