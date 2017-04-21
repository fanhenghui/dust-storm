#include "mi_ray_casting_cpu_brick_acc.h"

#include "boost/thread.hpp"

#include "MedImgCommon/mi_concurrency.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgArithmetic/mi_sampler.h"
#include "MedImgArithmetic/mi_ortho_camera.h"
#include "MedImgArithmetic/mi_matrix4.h"
#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"
#include "mi_mpr_entry_exit_points.h"
#include "mi_ray_caster_canvas.h"
#include "mi_brick_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

    namespace//TODO 这个函数要移到碰撞检测算法去
{
    //Return true if out
    bool check_outside(Vector3f pt, Vector3f bound_min , Vector3f bound_max)
    {
        if (pt._m[0] <= bound_min._m[0] || pt._m[1] <= bound_min._m[1] || pt._m[2] < bound_min._m[2]||
            pt._m[0] > bound_max._m[0] || pt._m[1] > bound_max._m[1] || pt._m[2] > bound_max._m[2])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //original
    //bool ray_intersect_aabb(Vector3f ray_start, Vector3f min, Vector3f bound, Vector3f ray_dir, 
    //    float& entry_step, float& exit_step)
    //{
    //    Vector3f bottom_step =  (min - ray_start) / ray_dir;
    //    Vector3f top_step = (min + bound - ray_start) / ray_dir;

    //    Vector3f min_step = min_per_elem(bottom_step, top_step);
    //    Vector3f max_step = max_per_elem(bottom_step, top_step);
    //    float near_step = min_step.max_elem();
    //    float far_step = max_step.min_elem();

    //    entry_step = std::max(near_step, 0.0f);
    //    exit_step = far_step;

    //    return entry_step < exit_step;
    //}
    //////////////////////////////////////////////////////////////////////////

    bool ray_intersect_aabb(Vector3f ray_start, Vector3f min, Vector3f bound, Vector3f ray_dir, 
        float& entry_step, float& exit_step)
    {
        Vector3f bottom_step =  (min - ray_start)/ray_dir;
        Vector3f top_step =  (min + bound - ray_start)/ray_dir;
        Vector3f bottom_step2(bottom_step);
        Vector3f top_step2(top_step);
        for (int i = 0 ; i< 3 ; ++i)
        {
            if (fabs(ray_dir._m[i]) <= FLOAT_EPSILON)
            {
                bottom_step._m[i] = -std::numeric_limits<float>::max();
                top_step._m[i] = -std::numeric_limits<float>::max();
                bottom_step2._m[i] = std::numeric_limits<float>::max();
                top_step2._m[i] = std::numeric_limits<float>::max();
            }
        }

        entry_step = bottom_step.min_per_elem(top_step).max_elem();
        exit_step = bottom_step2.max_per_elem(top_step2).min_elem();

        //////////////////////////////////////////////////////////////////////////
        //fNear > fFar not intersected
        //fNear >0  fFar > 0 fNear <= fFar intersected , start point not arrive AABB yet
        //fNear <0 fFar > 0 intersected , start point is in AABB
        //fNear <0 fFar < 0 fNear < fFar , intersected , but start point is outside AABB in extension ray 
        return entry_step < exit_step;
    }

    //If ray[i] < FLOAT_EPSLION then set ray[i] = 1 adjust[i] = std::numeric_limits<float>::max()*0.5f
    bool ray_intersect_aabb_acc(Vector3f ray_start, Vector3f min, Vector3f bound, Vector3f ray_dir, Vector3f vAdjust,
        float& entry_step, float& exit_step)
    {
        Vector3f bottom_step =  (min - ray_start)/ray_dir;
        Vector3f top_step =  (min + bound - ray_start)/ray_dir;
        Vector3f bottom_step2(bottom_step);
        Vector3f top_step2(top_step);
        bottom_step -= vAdjust;
        top_step -= vAdjust;
        bottom_step2 += vAdjust;
        top_step2 += vAdjust;

        entry_step = bottom_step.min_per_elem(top_step).max_elem();
        exit_step = bottom_step2.max_per_elem(top_step2).min_elem();

        //////////////////////////////////////////////////////////////////////////
        //fNear > fFar not intersected
        //fNear >0  fFar > 0 fNear <= fFar intersected , start point not arrive AABB yet
        //fNear <0 fFar > 0 intersected , start point is in AABB
        //fNear <0 fFar < 0 fNear < fFar , intersected , but start point is outside AABB in extension ray 
        return entry_step < exit_step;
    }
}

bool operator <(const BrickDistance& l , const BrickDistance& r)
{
    return l.distance < r.distance;
}

RayCastingCPUBrickAcc::RayCastingCPUBrickAcc(std::shared_ptr<RayCaster> ray_caster):
        _ray_caster(ray_caster),
        _width(32),
        _height(32),
        _entry_points(nullptr),
        _exit_points(nullptr),
        _canvas_array(nullptr),
        _brick_corner_array(nullptr),
        _volume_brick_unit_array(nullptr),
        _mask_brick_unit_array(nullptr),
        _volume_brick_info_array(nullptr),
        _mask_brick_info_array(nullptr),
        _brick_size(32),
        _brick_expand(2),
        _brick_count(0),
        _intersected_brick_num(0),
        _ray_count(0)
{
    _dim[0] = _dim[1] = _dim[2] = 32;
    _brick_dim[0] = _brick_dim[1] = _brick_dim[2] = 0;

    _test_pixel_x = 123546;
    _test_pixel_y = 123546;
}

RayCastingCPUBrickAcc::~RayCastingCPUBrickAcc()
{

}

void RayCastingCPUBrickAcc::render(int test_code /*= 0*/)
{
    try
    {
        std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster);

        //Volume info
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster->_entry_exit_points);
        ray_caster->_entry_exit_points->get_display_size(_width , _height);

        std::shared_ptr<ImageData> volume_img = ray_caster->_volume_data;
        RENDERALGO_CHECK_NULL_EXCEPTION(volume_img);
        memcpy(_dim , volume_img->_dim , sizeof(unsigned int)*3);

        //Brick struct
        _brick_size = ray_caster->_brick_size;
        _brick_expand = ray_caster->_brick_expand;
        _brick_corner_array = ray_caster->_brick_corner_array;
        _volume_brick_unit_array = ray_caster->_volume_brick_unit_array;
        _mask_brick_unit_array = ray_caster->_mask_brick_unit_array;
        _volume_brick_info_array = ray_caster->_volume_brick_info_array;
        _mask_brick_info_array = ray_caster->_mask_brick_info_array;
        unsigned int brick_dim[3] = {1,1,1};
        BrickUtils::instance()->get_brick_dim(_dim , brick_dim , _brick_size);
        _brick_count = brick_dim[0]*brick_dim[1]*brick_dim[2];
        if( !(_brick_dim[0] ==brick_dim[0] && _brick_dim[1] == brick_dim[1] && _brick_dim[2] == brick_dim[2]) )
        {
            memcpy(_brick_dim , brick_dim , sizeof(unsigned int)*3);
            _brick_center_distance.clear();
            _brick_center_distance.resize(_brick_count);
        }

        if (_ray_count != _width*_height)
        {
            _ray_count = _width*_height;
            _ray_result.reset(new float[_ray_count]);
        }
        //memset(_ray_result.get() , 0 , sizeof(float)*_ray_count);


        //Entry exit points
        _entry_points = ray_caster->_entry_exit_points->get_entry_points_array();
        _exit_points = ray_caster->_entry_exit_points->get_exit_points_array();

        //Canvas
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster->_canvas);
        _canvas_array = ray_caster->_canvas->get_color_array();
        RENDERALGO_CHECK_NULL_EXCEPTION(_canvas_array);
        memset(_canvas_array , 0 , sizeof(RGBAUnit)*_ray_count);
        //if ()
        {
            //TODO memset gray
        }

        //Matrix
        const Matrix4 mat_v2w = ray_caster->_mat_v2w;
        const Matrix4 mat_vp = ray_caster->_camera->get_view_projection_matrix();
        const Matrix4 mat_mvp = mat_vp*mat_v2w;
        const Matrix4 mat_mvp_inv = mat_mvp.get_inverse();
        _mat_mvp = ArithmeticUtils::convert_matrix(mat_mvp);
        _mat_mvp_inv = ArithmeticUtils::convert_matrix(mat_mvp_inv);
        _mat_mvp_inv_double = mat_mvp_inv;

        //////////////////////////////////////////////////////////////////////////
        //1 Brick sort
        clock_t t0 = clock();
        sort_brick_i();

        clock_t t1= clock();
        std::cout << "Sort brick cost : " << double(t1 - t0) << " ms.\n";

        //2 Brick ray casting
        for (unsigned int i = 0 ; i<_intersected_brick_num ; ++i)
        {
            ray_casting_in_brick_i(_brick_center_distance[i].id , ray_caster);
        }
        clock_t t2= clock();
        std::cout << "Ray casting cost : " << double(t2 - t1) << " ms.\n";


        //////////////////////////////////////////////////////////////////////////
        //3 update color to texture
        ray_caster->_canvas->update_color_array();

    }
    catch (const Exception& e)
    {
#ifdef _DEBUG
        //TODO LOG
        std::cout << e.what();
#endif
        assert(false);
        throw e;
    }
}

void RayCastingCPUBrickAcc::sort_brick_i()
{
    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    std::shared_ptr<MPREntryExitPoints> mpr_entry_exit_points = std::dynamic_pointer_cast<MPREntryExitPoints>(ray_caster->_entry_exit_points);
    if (mpr_entry_exit_points)//MPR
    {
        //1 Get bricks between entry and exit
        const Matrix4 mat_w2v = ray_caster->_mat_v2w.get_inverse();
        const Point3 eye_double = mat_w2v.transform(ray_caster->_camera->get_eye());
        const Vector3f eye((float)eye_double.x , (float)eye_double.y ,(float)eye_double.z);
        Vector3f center;
        const float brick_size_f = (float)_brick_size;
        Vector3f half_brick_bound(brick_size_f*0.5f);

        Vector4f entry_plane;
        Vector4f exit_plane;
        mpr_entry_exit_points->get_entry_exit_plane(entry_plane , exit_plane, _ray_dir_norm);
        Vector4f min , max ;
        Vector4f pt[8];
        const Vector4f brick_bound(brick_size_f ,brick_size_f ,brick_size_f , 0);
        _intersected_brick_num = 0;
        for (unsigned int i = 0 ; i < _brick_count ; ++i)
        {
            const BrickCorner &bc = _brick_corner_array[i];
            min = Vector4f((float)bc.min[0] , (float)bc.min[1] , (float)bc.min[2] , -1.0f);
            max = min + brick_bound;

            pt[0] = min;
            pt[1] = Vector4f(min._m[0],min._m[1],max._m[2], -1.0f);
            pt[1] =Vector4f(min._m[0],min._m[1],max._m[2], -1.0f);
            pt[2] =Vector4f(min._m[0],max._m[1],min._m[2], -1.0f);
            pt[3] =Vector4f(min._m[0],max._m[1],max._m[2], -1.0f);
            pt[4] =Vector4f(max._m[0],min._m[1],min._m[2], -1.0f);
            pt[5] =Vector4f(max._m[0],min._m[1],max._m[2], -1.0f);
            pt[6] =Vector4f(max._m[0],max._m[1],min._m[2], -1.0f);
            pt[7] = max;

            //Intersect with entry plane
            int entry_positive = 0;
            int entry_negative = 0;
            for ( int j = 0 ; j<8 ; ++j)
            {
                if (pt[j].dot_product(entry_plane) > 0.0f)
                {
                    entry_positive += 1;
                }
                else
                {
                    entry_negative +=1;
                }
            }

            if (8 != entry_positive && 8 != entry_negative)//intersection
            {
                center = Vector3f(min._m128) + half_brick_bound;

                _brick_center_distance[_intersected_brick_num].id = i;
                _brick_center_distance[_intersected_brick_num].distance = (center - eye).dot_product(entry_plane.get_128());
                ++_intersected_brick_num;
                continue;
            }

            int exit_positive = 0;
            int exit_negative = 0;
            for ( int j = 0 ; j<8 ; ++j)
            {
                if (pt[j].dot_product(exit_plane) > 0.0f)
                {
                    exit_positive += 1;
                }
                else
                {
                    exit_negative +=1;
                }
            }

            if (8 != exit_positive && 8 != exit_negative)//intersection
            {
                center = Vector3f(min._m128) + half_brick_bound;

                _brick_center_distance[_intersected_brick_num].id = i;
                _brick_center_distance[_intersected_brick_num].distance = (center - eye).dot_product(entry_plane.get_128());
                ++_intersected_brick_num;
                continue;
            }

            if (8 == entry_positive && 8 == exit_positive)
            {
                center = Vector3f(min._m128) + half_brick_bound;

                _brick_center_distance[_intersected_brick_num].id = i;
                _brick_center_distance[_intersected_brick_num].distance = (center - eye).dot_product(entry_plane.get_128());
                ++_intersected_brick_num;
                continue;
            }
        }

        std::cout << "Intersect brick count : " << _intersected_brick_num << std::endl;

        //2 Sort bricks
        if (_intersected_brick_num > 1)
        {
            std::sort(_brick_center_distance.begin() , _brick_center_distance.begin() + _intersected_brick_num , std::less<BrickDistance>());
        }

    }
    else//Maybe VR
    {
        RENDERALGO_THROW_EXCEPTION("Entry exit points is not MPR!");
    }
}
const std::vector<BrickDistance>& RayCastingCPUBrickAcc::get_brick_distance() const
{
    return _brick_center_distance;
}

unsigned int RayCastingCPUBrickAcc::get_ray_casting_brick_count() const
{
    return _intersected_brick_num;
}

void RayCastingCPUBrickAcc::ray_casting_in_brick_i(unsigned int brick_id ,  const std::shared_ptr<RayCaster>& ray_caster)
{
    //1 inverse projection
    const BrickCorner &bc = _brick_corner_array[brick_id];
    const Vector3f brick_bound((float)_brick_size);
    //const Vector3f vBrickExpand((float)_brick_expand);
    const Vector3f min = Vector3f((float)bc.min[0] , (float)bc.min[1] , (float)bc.min[2]);
    const Vector3f max = min + brick_bound;
    //const unsigned int uiBrickSampleSize = _brick_expand + _brick_size;

    const Vector3f pt[8] = {min , 
        Vector3f(min._m[0],min._m[1],max._m[2]),
        Vector3f(min._m[0],max._m[1],min._m[2]),
        Vector3f(min._m[0],max._m[1],max._m[2]),
        Vector3f(max._m[0],min._m[1],min._m[2]),
        Vector3f(max._m[0],min._m[1],max._m[2]),
        Vector3f(max._m[0],max._m[1],min._m[2]),
        max};

    const float width_float = (float)_width;
    const float height_float = (float)_height;
    int x_begin(65535), x_end(-65535) , y_begin(65535) , y_end(-65535);
    int current_x , current_y;
    float x_norm , nrom_y;
    Vector3f screen_point;
    int out_tag = 1;//completely outside of screen
    int brick_vertex_out = 0;
    for (int i = 0 ; i< 8 ; ++i)
    {
        screen_point = _mat_mvp.transform_point(pt[i]);
        x_norm = screen_point._m[0];
        nrom_y = screen_point._m[1];

        out_tag = 1;
        if (x_norm < -1)
        {
            x_norm = -1;
            out_tag+=1;
        }
        else if (x_norm > 1)
        {
            x_norm = 1;
            out_tag+=1;
        }

        if (nrom_y < -1)
        {
            nrom_y = -1;
            out_tag+=1;
        }
        else if (nrom_y > 1)
        {
            nrom_y = 1;
            out_tag+=1;
        }

        brick_vertex_out += (out_tag>>1);


        //if (fNormX < -1 || fNormX > 1||
        //    fNormY < -1 || fNormY > 1)
        //{
        //    iBrickVertexOut +=1;
        //    continue;
        //}

        current_x = int( (x_norm+1.0f)*0.5f * width_float + 0.5);
        if (current_x < x_begin)
        {
            x_begin = current_x;
        }
        if (current_x > x_end)
        {
            x_end = current_x;
        }

        current_y = int( (nrom_y+1.0f)*0.5f*height_float + 0.5);
        if (current_y < y_begin)
        {
            y_begin = current_y;
        }
        if (current_y > y_end)
        {
            y_end = current_y;
        }
    }

    if (y_end > _height || x_end > _width)
    {
        //std::cout <<"ERR\n";
    }

    if (brick_vertex_out < 8)
    {
        //Multi-thread
        const int x_range = x_end - x_begin;
        const int y_range = y_end - y_begin;
        const int pixel_num = x_range*y_range;
        int quad_range[4] = {x_begin ,x_range , y_begin , y_range};


        const Vector3f brick_bound((float)_brick_size);
        const Vector3f brick_expand((float)_brick_expand);
        const unsigned int brick_sample_size = _brick_size + _brick_expand*2;
        const Vector3f ray_dir_sample(_ray_dir_norm*ray_caster->_sample_rate);
        Sampler<unsigned short> sampler;
        unsigned short* data_array = (unsigned short*)_volume_brick_unit_array[brick_id].data;

        //////////////////////////////////////////////////////////////////////////
        //Adjust ray direction
        Vector3f ray_brick(ray_dir_sample);
        Vector3f ray_brick_adjust(0,0,0);
        for (int i = 0 ; i< 3 ; ++i)
        {
            if (fabs(ray_brick._m[i]) <= FLOAT_EPSILON)
            {
                ray_brick._m[i] = 1;//be divided
                ray_brick_adjust._m[i] = std::numeric_limits<float>::max()*0.5f;
            }
        }
        //////////////////////////////////////////////////////////////////////////

        const float min_wl_gray = ray_caster->_global_wl - ray_caster->_global_ww*0.5f;
        const float ww_r = 1.0f/ray_caster->_global_ww;

#pragma omp parallel for 
        for (int pixel_id = 0; pixel_id < pixel_num ; ++pixel_id)
        {
            int y = pixel_id / x_range;
            int x = pixel_id - y*x_range;
            x += x_begin;
            y += y_begin;
            const int ray_id = y*_width + x;
            if (x == _test_pixel_x && y == _test_pixel_y)
            {
                std::cout <<"ERR";
            }

            float last_step , end_step , current_brick_step;
            Vector3f start_point;
            float entry_step , exit_step;
            Vector3f sample_pos;
            float max_gray = -65535.0f;
            float sample_value , gray;

            Vector4f &entry_point = _entry_points[ray_id];//Use entry points array to store ray parameter
            last_step = entry_point._m[3];
            end_step = _exit_points[ray_id]._m[3];

            if (last_step >= end_step) //brick ray casting end
            {
                continue;
            }

            if (last_step < -0.5f) //Skip points which decided in entry points calculation
            {
                _canvas_array[ray_id] =RGBAUnit();
                //const bool bIntersection = RayIntersectAABB(ptEntryF, Vector3f(0,0,0), vDim, vRayDir, entry_step, exit_step);
            }
            else
            {

                //1 Get entry exit points
                start_point._m128 = (_entry_points[ray_id]._m128);

                if (last_step < FLOAT_EPSILON)//Zero step, the first step
                {
                    _ray_result[ray_id] = -65535.0f;//initialize max gray
                }
                else
                {
                    start_point += ray_dir_sample*(last_step + 1.0f);//Current point , step forward once
                }

                if(check_outside(start_point , min , min+brick_bound))
                {
                    continue;
                }

                if (ray_intersect_aabb_acc(start_point, min, brick_bound, ray_brick, ray_brick_adjust , entry_step, exit_step))
                {
                    max_gray = -65535.0f;
                    sample_value = 0.0f;

                    current_brick_step = (float)(int(exit_step - entry_step + 0.5f));
                    if (current_brick_step + last_step > end_step)
                    {
                        current_brick_step = end_step - last_step;
                    }

                    for (float fSampleStep = 0.0f ; fSampleStep < current_brick_step-0.001f ; fSampleStep+=1.0f)
                    {
                        sample_pos = start_point + ray_dir_sample*fSampleStep;
                        sample_pos = sample_pos - min + brick_expand; 
                        sample_value = sampler.sample_3d_linear(sample_pos._m[0] , sample_pos._m[1] , sample_pos._m[2] , 
                            brick_sample_size , brick_sample_size , brick_sample_size , data_array);
                        max_gray = sample_value > max_gray ?  sample_value : max_gray;
                    }

                    if (max_gray > _ray_result[ray_id])
                    {
                        _ray_result[ray_id] = max_gray;
                        gray = (max_gray - min_wl_gray)*ww_r;
                        _canvas_array[ray_id] = RGBAUnit(gray , gray , gray);
                    }

                    entry_point._m[3] = last_step + current_brick_step;
                }
            }
        }

    }
}




MED_IMAGING_END_NAMESPACE