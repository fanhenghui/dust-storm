#ifndef MED_IMAGING_RAY_CASTING_BRICK_ACC_H_
#define MED_IMAGING_RAY_CASTING_BRICK_ACC_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgArithmetic/mi_vector4f.h"
#include "MedImgArithmetic/mi_sampler.h"
#include "MedImgArithmetic/mi_matrix4f.h"
#include "MedImgArithmetic/mi_matrix4.h"

MED_IMAGING_BEGIN_NAMESPACE

class RayCaster;

class RayCastingCPUBrickAcc
{
public:
    RayCastingCPUBrickAcc(std::shared_ptr<RayCaster> ray_caster);
    ~RayCastingCPUBrickAcc();

    void render(int test_code = 0);

    //////////////////////////////////////////////////////////////////////////
    //For testing 
    const std::vector<BrickDistance>& get_brick_distance() const;
    unsigned int get_ray_casting_brick_count() const;
public:

private:
    void sort_brick_i();//Just for orthogonal camera(same ray direction)

    void ray_casting_in_brick_i(unsigned int id ,  const std::shared_ptr<RayCaster>& ray_caster);

private:
    std::weak_ptr<RayCaster> _ray_caster;
    //Cache
    int _width;
    int _height;
    Vector4f* _entry_points;
    Vector4f* _exit_points;
    unsigned int _dim[3];
    RGBAUnit* _canvas_array;

    //Brick struct
    unsigned int _brick_dim[3];
    unsigned int _brick_size;
    unsigned int _brick_expand;
    unsigned int _brick_count;
    BrickCorner* _brick_corner_array;
    BrickUnit* _volume_brick_unit_array;
    VolumeBrickInfo* _volume_brick_info_array;
    BrickUnit* _mask_brick_unit_array;
    MaskBrickInfo* _mask_brick_info_array;
    Matrix4f _mat_mvp;
    Matrix4f _mat_mvp_inv;
    Matrix4 _mat_mvp_inv_double;


    //Brick cache
    std::vector<BrickDistance> _brick_center_distance;
    unsigned int _intersected_brick_num;
    std::unique_ptr<float[]> _ray_result;
    int _ray_count;
    Vector3f _ray_dir_norm;
    //////////////////////////////////////////////////////////////////////////
    //Test 
    int _test_pixel_x;
    int _test_pixel_y;

};


MED_IMAGING_END_NAMESPACE



#endif