#include "mi_ray_casting_cpu.h"

#include "boost/thread.hpp"

#include "MedImgIO/mi_image_data.h"
#include "MedImgCommon/mi_concurrency.h"
#include "MedImgArithmetic/mi_sampler.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"
#include "mi_ray_caster_canvas.h"

MED_IMAGING_BEGIN_NAMESPACE

RayCastingCPU::RayCastingCPU(std::shared_ptr<RayCaster> ray_caster):
        _ray_caster(ray_caster),
        _width(32),
        _height(32),
        _entry_points(nullptr),
        _exit_points(nullptr),
        _volume_data_array(nullptr),
        _mask_data_array(nullptr),
        _canvas_array(nullptr)
{
    _dim[0] = _dim[1] = _dim[2] = 32;
}

RayCastingCPU::~RayCastingCPU()
{

}

void RayCastingCPU::render(int test_code )
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
        _volume_data_array = volume_img->get_pixel_pointer();

        //Entry exit points
        _entry_points = ray_caster->_entry_exit_points->get_entry_points_array();
        _exit_points = ray_caster->_entry_exit_points->get_exit_points_array();

        //Canvas
        RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster->_canvas);
        _canvas_array = ray_caster->_canvas->get_color_array();
        RENDERALGO_CHECK_NULL_EXCEPTION(_canvas_array);

        //////////////////////////////////////////////////////////////////////////
        //For testing entry & exit points
        if (0 != test_code)
        {
           render_entry_exit_points_i(test_code);
            ray_caster->_canvas->update_color_array();
            return;
        }
        //////////////////////////////////////////////////////////////////////////



        switch(volume_img->_data_type)
        {
        case USHORT:
            {
                ray_casting_i<unsigned short>( ray_caster );
                break;
            }

        case SHORT:
            {
                ray_casting_i<short>( ray_caster );
                break;
            }

        case FLOAT:
            {
                ray_casting_i<float>( ray_caster );
                break;
            }
        default:
            RENDERALGO_THROW_EXCEPTION("Undefined data type!");
        }

        CHECK_GL_ERROR;

        if (COMPOSITE_AVERAGE == ray_caster->_composite_mode ||
            COMPOSITE_MIP == ray_caster->_composite_mode ||
            COMPOSITE_MINIP == ray_caster->_composite_mode)
        {
            ray_caster->_canvas->update_color_array();
        }

        CHECK_GL_ERROR;

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
    catch (const std::exception& e)
    {
#ifdef _DEBUG
        //TODO LOG
        std::cout << e.what();
#endif
        assert(false);
        throw e;
    }
}


template<class T>
void RayCastingCPU::ray_casting_i(std::shared_ptr<RayCaster> ray_caster )
{
    switch(ray_caster->_composite_mode)
    {
    case COMPOSITE_AVERAGE:
        {
            ray_casting_average_i<T>(ray_caster);
            break;
        }
    case COMPOSITE_MIP:
        {
            ray_casting_mip_i<T>(ray_caster );
            break;
        }
    case COMPOSITE_MINIP:
        {
            ray_casting_minip_i<T>(ray_caster);
            break;
        }
    default:
        break;
    }
}

template<class T>
void RayCastingCPU::ray_casting_average_i(std::shared_ptr<RayCaster> ray_caster)
{
    const Sampler<T> sampler;
    const int pixel_sum = _width*_height;

#ifndef _DEBUG
#pragma omp parallel for
#endif
    for (int idx = 0; idx<pixel_sum  ; ++idx)
    {
        const int y = idx / _width;
        const int x = idx - y*_width;

        //1 Get entry exit points
        const Vector3f start_point(_entry_points[idx]._m128);
        const Vector3f end_point(_exit_points[idx]._m128);

        const bool skip = start_point._m[3] < -0.5f; // -1.0 for skip , 0  for valid entry exit points
        if (skip)
        {
            _canvas_array[idx] = RGBAUnit();
            continue;
        }

        const Vector3f dir = end_point - start_point;
        const float length = dir.magnitude();
        const Vector3f dir_step = dir.get_normalize()*ray_caster->_sample_rate;
        const float step_float = length / ray_caster->_sample_rate;
        int step =(int)step_float;
        if (step == 0)//保证至少积分一次
        {
            step = 1;
        }

        //2 Integrate
        const float ratio =1000.0f;
        const float ratio_r = 1.0f/1000.0f;
        float sum = 0.0f;
        Vector3f sample_pos = start_point;

        float sample_value = 0.0f;
        for (int i = 0 ; i < step ; ++i)
        {
            sample_pos += ( dir_step * float(i) );

            sample_value = sampler.sample_3d_linear(
                sample_pos._m[0] , sample_pos._m[1] , sample_pos._m[2] , 
                _dim[0], _dim[1], _dim[2],
                (T*)_volume_data_array);

            sum += sample_value*ratio_r;
        }
        const float result_gray  = sum *(1.0f/step) * ratio;

        //3Apply window level
        const float min_wl_gray = ray_caster->_global_wl - ray_caster->_global_ww*0.5f;
        const float gray = (result_gray - min_wl_gray)/ray_caster->_global_ww;

        //4Apply pseudo color
        //TODO just gray
        _canvas_array[idx] = RGBAUnit(gray , gray , gray);
    }

}

template<class T>
void RayCastingCPU::ray_casting_mip_i( std::shared_ptr<RayCaster> ray_caster)
{
    const Sampler<T> sampler;
    const int pixel_sum = _width*_height;

#pragma omp parallel for
    for (int idx = 0; idx<pixel_sum  ; ++idx)
    {
        const int y = idx / _width;
        const int x = idx - y*_width;

        //1 Get entry exit points
        const Vector3f start_point(_entry_points[idx]._m128);
        const Vector3f end_point(_exit_points[idx]._m128);

        const bool skip = start_point._m[3] <  -0.5f; // -1.0 for skip , 0  for valid entry exit points
        if (skip)
        {
            _canvas_array[idx] = RGBAUnit();
            continue;
        }

        const Vector3f dir = end_point - start_point;
        const float length = dir.magnitude();
        const Vector3f dir_step = dir.get_normalize()*ray_caster->_sample_rate;
        const float step_float = length / ray_caster->_sample_rate;
        int step =(int)step_float;
        if (step == 0)//保证至少积分一次
        {
            step = 1;
        }

        //2 Integrate
        float max_gray = -65535.0f;
        Vector3f sample_pos = start_point;

        float sample_value = 0.0f;
        for (int i = 0 ; i < step ; ++i)
        {
            sample_pos += ( dir_step * float(i) );

            sample_value = sampler.sample_3d_linear(
                sample_pos._m[0] , sample_pos._m[1] , sample_pos._m[2] , 
                _dim[0], _dim[1], _dim[2],
                (T*)_volume_data_array);

            max_gray = sample_value > max_gray ? sample_value : max_gray;
        }

        //3Apply window level
        const float min_wl_gray = ray_caster->_global_wl - ray_caster->_global_ww*0.5f;
        const float gray = (max_gray - min_wl_gray)/ray_caster->_global_ww;

        //4Apply pseudo color
        //TODO just gray
        _canvas_array[idx] = RGBAUnit(gray , gray , gray);
    }
}

template<class T>
void RayCastingCPU::ray_casting_minip_i( std::shared_ptr<RayCaster> ray_caster)
{
    const Sampler<T> sampler;
    const int pixel_sum = _width*_height;

#pragma omp parallel for
    for (int idx = 0; idx<pixel_sum  ; ++idx)
    {
        const int y = idx / _width;
        const int x = idx - y*_width;

        //1 Get entry exit points
        const Vector3f start_point(_entry_points[idx]._m128);
        const Vector3f end_point(_exit_points[idx]._m128);

        const bool skip = start_point._m[3] < -0.5f; // -1.0 for skip , 0  for valid entry exit points
        if (skip)
        {
            _canvas_array[idx] = RGBAUnit();
            continue;
        }

        const Vector3f dir = end_point - start_point;
        const float length = dir.magnitude();
        const Vector3f dir_step = dir.get_normalize()*ray_caster->_sample_rate;
        const float step_float = length / ray_caster->_sample_rate;
        int step =(int)step_float;
        if (step == 0)//保证至少积分一次
        {
            step = 1;
        }

        //2 Integrate
        float max_gray = -65535.0f;
        Vector3f sample_pos = start_point;

        float sample_value = 0.0f;
        for (int i = 0 ; i < step ; ++i)
        {
            sample_pos += ( dir_step * float(i) );

            sample_value = sampler.sample_3d_linear(
                sample_pos._m[0] , sample_pos._m[1] , sample_pos._m[2] , 
                _dim[0], _dim[1], _dim[2],
                (T*)_volume_data_array);

            max_gray = sample_value > max_gray ? sample_value : max_gray;
        }

        //3Apply window level
        const float min_wl_gray = ray_caster->_global_wl - ray_caster->_global_ww*0.5f;
        const float gray = (max_gray - min_wl_gray)/ray_caster->_global_ww;

        //4Apply pseudo color
        //TODO just gray
        _canvas_array[idx] = RGBAUnit(gray , gray , gray);
    }
}

void RayCastingCPU::render_entry_exit_points_i( int test_code)
{
    Vector3f vDimR(1.0f/_dim[0] , 1.0f/_dim[1] , 1.0f/_dim[2]);
    const int pixel_sum = _width*_height;
    if (1 == test_code)
    {
        for (int i = 0 ; i < pixel_sum ; ++i)
        {
            Vector3f start_point(_entry_points[i]._m128);
            start_point =start_point*vDimR;
            _canvas_array[i] = RGBAUnit(start_point._m[0] , start_point._m[1] , start_point._m[2]);
        }
    }
    else
    {
        for (int i = 0 ; i < pixel_sum ; ++i)
        {
            Vector3f end_point(_exit_points[i]._m128);
            end_point =end_point*vDimR;
            _canvas_array[i] = RGBAUnit(end_point._m[0] , end_point._m[1] , end_point._m[2]);
        }
    }


}


MED_IMAGING_END_NAMESPACE