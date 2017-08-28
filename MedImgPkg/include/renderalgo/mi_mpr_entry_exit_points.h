#ifndef MEDIMGRENDERALGO_ENTRY_EXIT_POINTS_H
#define MEDIMGRENDERALGO_ENTRY_EXIT_POINTS_H

#include "arithmetic/mi_vector3f.h"
#include "arithmetic/mi_vector4f.h"
#include "glresource/mi_gl_resource_define.h"
#include "io/mi_io_define.h"
#include "renderalgo/mi_entry_exit_points.h"
#include "renderalgo/mi_ray_caster_define.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export MPREntryExitPoints : public EntryExitPoints {
public:
  MPREntryExitPoints();

  virtual ~MPREntryExitPoints();

  virtual void initialize();

  void set_thickness(float thickness); // In volume coordinate

  void set_sample_rate(float sample_rate);

  virtual void calculate_entry_exit_points();

  //////////////////////////////////////////////////////////////////////////
  /// \ plane function :  ax + by + cz = d
  /// \ Or normal&point x*N = d , thus d is the distance between original(0,0,0)
  /// and the plane
  /// \ vector4f (a,b,c,d)
  ///\ Entry plane's normal is form entry to exit
  ///\ Exit plane's normal is form exit to entry
  /// \In volume coordinate
  //////////////////////////////////////////////////////////////////////////
  void get_entry_exit_plane(Vector4f &entry_point, Vector4f &exit_point,
                            Vector3f &ray_dir_norm);

private:
  void cal_entry_exit_points_cpu_i();

  void cal_entry_exit_points_gpu_i();

private:
  float _thickness;
  float _sample_rate;

  // Entry exit plane(orthogonal)
  Vector4f _entry_plane;
  Vector4f _exit_plane;
  Vector3f _ray_dir_norm;

  float _standard_steps;

  // GPU entry exit points cal
  GLProgramPtr _program;
};

MED_IMG_END_NAMESPACE

#endif