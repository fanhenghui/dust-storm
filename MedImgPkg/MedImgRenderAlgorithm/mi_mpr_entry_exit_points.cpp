#include "mi_mpr_entry_exit_points.h"

#include <limits>
#include <time.h>
#include "boost/thread.hpp"

#include "MedImgIO/mi_image_data.h"
#include "MedImgArithmetic/mi_camera_base.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_shader_collection.h"
#include "mi_camera_calculator.h"

MED_IMG_BEGIN_NAMESPACE

namespace
{
    //Return true if out
    bool check_outside(Vector3f pt, Vector3f bound)
    {
        if (pt._m[0] <0 || pt._m[1] < 0 || pt._m[2] < 0 ||
            pt._m[0] > bound._m[0] || pt._m[1] > bound._m[1] || pt._m[2] > bound._m[2])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    //If ray[i] < FLOAT_EPSLION then set ray[i] = 1 adjust[i] = std::numeric_limits<float>::max()*0.5f
    bool ray_intersect_aabb_acc(Vector3f ray_start, Vector3f min, Vector3f bound, Vector3f ray_norm, Vector3f adjust,
        float& entry_step, float& exit_step)
    {
        Vector3f bottom_step =  (min - ray_start)/ray_norm;
        Vector3f top_step =  (min + bound - ray_start)/ray_norm;
        Vector3f bottom_step2(bottom_step);
        Vector3f top_step2(top_step);
        bottom_step -= adjust;
        top_step -= adjust;
        bottom_step2 += adjust;
        top_step2 += adjust;

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

MPREntryExitPoints::MPREntryExitPoints():_thickness(1.0f),_entry_plane(1,0,0,0),_exit_plane(1,0,0,0),_sample_rate(1.0)
{

}

MPREntryExitPoints::~MPREntryExitPoints()
{

}

void MPREntryExitPoints::set_sample_rate(float sample_rate)
{
    _sample_rate = sample_rate;
}

void MPREntryExitPoints::set_thickness(float thickness)
{
    _thickness = thickness;
}

void MPREntryExitPoints::calculate_entry_exit_points()
{
    _standard_steps = float(int(_thickness / _sample_rate + 0.5f));

    //clock_t t0 = clock();
    if (CPU_BASE == _strategy)
    {
        cal_entry_exit_points_cpu_i();
    }
    else if (CPU_BRICK_ACCELERATE == _strategy)
    {
        cal_entry_exit_points_cpu_i();
        cal_entry_exit_plane_cpu_i();
    }
    else if (GPU_BASE == _strategy)
    {
        cal_entry_exit_points_gpu_i();
    }
    //clock_t t1 = clock();
    //std::cout << "Calculate entry exit points cost : " << double(t1 - t0) << std::endl;
}

void MPREntryExitPoints::cal_entry_exit_points_cpu_i()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(_camera);
        RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume_data);

        const Vector3f dim_vector((float)_volume_data->_dim[0],
            (float)_volume_data->_dim[1],
            (float)_volume_data->_dim[2]);

        //Calculate base plane of MPR
        const Matrix4 mat_v2w = _camera_calculator->get_volume_to_world_matrix();
        const Matrix4 mat_vp = _camera->get_view_projection_matrix();
        const Matrix4 mat_mvp = mat_vp*mat_v2w;
        const Matrix4 mat_mvp_inv = mat_mvp.get_inverse();

        /*const Point3 pt00 = mat_mvp_inv.transform(Point3(-1.0,-1.0,0));
        const Point3 pt01 = mat_mvp_inv.transform(Point3(-1.0,1.0,0));
        const Point3 pt10 = mat_mvp_inv.transform(Point3(1.0,-1.0,0));*/

        Point2 pt00_2 = ArithmeticUtils::dc_to_ndc(Point2(0,_height - 1) , _width , _height);
        Point2 pt01_2 = ArithmeticUtils::dc_to_ndc(Point2(0,0) , _width , _height) ;
        Point2 pt10_2 = ArithmeticUtils::dc_to_ndc(Point2(_width-1,_height - 1) , _width , _height);
        const Point3 pt00 = mat_mvp_inv.transform(Point3( pt00_2.x , pt00_2.y , 0) );
        const Point3 pt01 = mat_mvp_inv.transform(Point3( pt01_2.x , pt01_2.y , 0) );
        const Point3 pt10 = mat_mvp_inv.transform(Point3( pt10_2.x , pt10_2.y , 0) );

        const Vector3 x_delta = (pt10 - pt00) * (1.0/(_width-1));
        const Vector3 y_delta = (pt01 - pt00) * (1.0/(_height-1));

        Vector3 view_dir = _camera->get_look_at() - _camera->get_eye();
        view_dir = mat_v2w.get_transpose().transform(view_dir);
        view_dir.normalize();

        const Vector3f ray_dir = ArithmeticUtils::convert_vector(view_dir);

        const Vector3f pt00F((float)pt00.x , (float)pt00.y , (float)pt00.z);
        const Vector3f x_delta_float((float)x_delta.x,(float)x_delta.y,(float)x_delta.z);
        const Vector3f y_delta_float((float)y_delta.x,(float)y_delta.y,(float)y_delta.z);

        const float thickness = _thickness;
        const float fThicknessHalf = thickness*0.5f;
        Vector4f* pEntryPoints = _entry_points_buffer.get();
        Vector4f* pExitPoints = _exit_points_buffer.get();

        //////////////////////////////////////////////////////////////////////////
        //Adjust ray direction
        Vector3f vRayBrick(ray_dir);
        Vector3f vRayBrickAdjust(0,0,0);
        for (int i = 0 ; i< 3 ; ++i)
        {
            if (fabs(vRayBrick._m[i]) <= FLOAT_EPSILON)
            {
                vRayBrick._m[i] = 1;//be divided
                vRayBrickAdjust._m[i] = std::numeric_limits<float>::max()*0.5f;
            }
        }
        //////////////////////////////////////////////////////////////////////////

        const int pixel_sum = _width*_height;
#ifndef _DEBUG
#pragma omp parallel for
#endif
        for (int idx = 0 ; idx< pixel_sum; ++idx)
        {
            Vector3f ptCurF;
            Vector3f ptEntryF;
            Vector3f ptExitF;
            Vector3f ptEntryIntersection;
            Vector3f ptExitIntersection;

            int iY = idx / _width;
            int iX = idx - iY*_width;

            ptCurF = pt00F + x_delta_float*(float)iX + y_delta_float*(float)iY;
            if (fThicknessHalf <= 1.0)
            {
                ptEntryF = ptCurF;
                ptExitF = ptCurF + ray_dir*fThicknessHalf*2;
            }
            else
            {
                ptEntryF = ptCurF - ray_dir*fThicknessHalf;
                ptExitF = ptCurF + ray_dir*fThicknessHalf;
            }

            ptEntryIntersection = ptEntryF;
            ptExitIntersection = ptExitF;

            //Intersect volume AABB to get intersected entry&exit points
            float entry_step(0), exit_step(0);
            const bool bIntersection = ray_intersect_aabb_acc(ptEntryF, Vector3f(0,0,0), dim_vector, vRayBrick, vRayBrickAdjust , entry_step, exit_step);

            //Entry point outside
            if( check_outside(ptEntryF, dim_vector) )
            {
                if(!bIntersection || entry_step < 0 || entry_step > thickness ) // check entry points in range of thickness and volume
                {
                    pEntryPoints[idx] = Vector4f(0,0,0,-1.0f);
                    pExitPoints[idx] = Vector4f(0,0,0,-1.0f);
                    continue;
                }
                ptEntryIntersection = ptEntryF + ray_dir*entry_step;
            }

            //Exit point outside
            if( check_outside(ptExitF, dim_vector) )
            {
                if(!bIntersection)
                {
                    pEntryPoints[idx] = Vector4f(0,0,0,-1.0f);
                    pExitPoints[idx] = Vector4f(0,0,0,-1.0f);
                    continue;
                }
                ptExitIntersection= ptEntryF + ray_dir*exit_step;
            }

            //////////////////////////////////////////////////////////////////////////
            //alpha value : ray step
            float fStep = (float)(int)( (ptExitIntersection - ptEntryIntersection).magnitude()/_sample_rate + 0.5f);
            if (fStep > _standard_steps)//Adjust step to prevent  fStep = standard step + epsilon which it's ceil equals ( standard cell + 1)
            {
                fStep = _standard_steps;
            }
            pEntryPoints[idx] = Vector4f(ptEntryIntersection,0.0f);//Entry step is 0 , the first sample position is on entry plane
            pExitPoints[idx] = Vector4f(ptExitIntersection,fStep);//Exit step is integer step which represent the integeration path

            //////////////////////////////////////////////////////////////////////////
        }

        /*initialize();
        _entry_points_texture->bind();
        _entry_points_texture->load(GL_RGBA32F , _width , _height , GL_RGBA , GL_FLOAT , _entry_points_buffer.get());

        _exit_points_texture->bind();
        _exit_points_texture->load(GL_RGBA32F , _width , _height , GL_RGBA , GL_FLOAT , _exit_points_buffer.get());

        _entry_points_texture->bind();
        _entry_points_texture->download(GL_RGBA , GL_FLOAT , _entry_points_buffer.get());

        _exit_points_texture->bind();
        _exit_points_texture->download(GL_RGBA , GL_FLOAT , _exit_points_buffer.get());*/
    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::cal_entry_exit_plane_cpu_i()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(_camera);
        RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume_data);

        Vector3f vDim((float)_volume_data->_dim[0],
            (float)_volume_data->_dim[1],
            (float)_volume_data->_dim[2]);

        //Calculate base plane of MPR
        const Matrix4 mat_v2w = _camera_calculator->get_volume_to_world_matrix();
        const Matrix4 mat_vp = _camera->get_view_projection_matrix();
        const Matrix4 mat_mvp = mat_vp*mat_v2w;
        const Matrix4 mat_mvp_inv = mat_mvp.get_inverse();

        Vector3 view_dir = _camera->get_look_at() - _camera->get_eye();
        view_dir = mat_v2w.get_transpose().transform(view_dir);
        view_dir.normalize();
        const Vector3 ray_dir = view_dir;
        _ray_dir_norm = ArithmeticUtils::convert_vector(ray_dir);

        const float thickness = _thickness;
        const float thickness_half = thickness*0.5f;

        const Point3 ptCenter = mat_mvp_inv.transform(Point3(0.0,0.0,0));
        const Point3 ptEntry = ptCenter - ray_dir*thickness_half;
        const Point3 ptExit = ptCenter + ray_dir*thickness_half;

        double dDisEntry = ray_dir.dot_product(ptEntry - Point3::S_ZERO_POINT);
        double dDisExit = (-ray_dir).dot_product(ptExit - Point3::S_ZERO_POINT);

        _entry_plane = Vector4f((float)ray_dir.x ,(float)ray_dir.y , (float)ray_dir.z , (float)dDisEntry);
        _exit_plane = Vector4f(-(float)ray_dir.x ,-(float)ray_dir.y , -(float)ray_dir.z , (float)dDisExit);

    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::get_entry_exit_plane(Vector4f& entry_point , Vector4f& exit_point , Vector3f& ray_dir_norm)
{
    entry_point = _entry_plane;
    exit_point = _exit_plane;
    ray_dir_norm = _ray_dir_norm;
}

void MPREntryExitPoints::cal_entry_exit_points_gpu_i()
{
    try
    {
#define IMAGE_ENTRY_POINT 0
#define IMAGE_EXIT_POINT 1
#define DISPLAY_SIZE 2
#define VOLUME_DIM 3
#define MVP_INVERSE 4
#define THICKNESS 5
#define RAY_DIRECTION 6

        CHECK_GL_ERROR;

        initialize();

        const unsigned int uiProgram = _program->get_id();
        if (0 ==uiProgram)
        {
            RENDERALGO_THROW_EXCEPTION("Program ID is 0!");
        }

        CHECK_GL_ERROR;

        _program->bind();

        glPushAttrib(GL_ALL_ATTRIB_BITS);

        CHECK_GL_ERROR;

        _entry_points_texture->bind_image(IMAGE_ENTRY_POINT , 0 , false , 0 , GL_READ_WRITE , GL_RGBA32F);
        _exit_points_texture->bind_image(IMAGE_EXIT_POINT , 0 , false , 0 , GL_READ_WRITE , GL_RGBA32F);

        CHECK_GL_ERROR;

        glProgramUniform2ui(uiProgram , DISPLAY_SIZE , (GLuint)_width , (GLuint)_height);

        CHECK_GL_ERROR;

        const float fDim[3] = {(float)_volume_data->_dim[0] , (float)_volume_data->_dim[1] , (float)_volume_data->_dim[2]};
        glProgramUniform3f(uiProgram , VOLUME_DIM , fDim[0] , fDim[1] , fDim[2]);

        CHECK_GL_ERROR;

        const Matrix4 mat_v2w = _camera_calculator->get_volume_to_world_matrix();
        const Matrix4 mat_vp = _camera->get_view_projection_matrix();
        const Matrix4 mat_mvp = mat_vp*mat_v2w;
        const Matrix4 mat_mvp_inv = mat_mvp.get_inverse();

        CHECK_GL_ERROR;

        float fMat[16] = {0.0f};
        mat_mvp_inv.to_float16(fMat);
        glProgramUniformMatrix4fv(uiProgram , MVP_INVERSE , 1 , GL_FALSE , fMat);

        CHECK_GL_ERROR;

        glProgramUniform1f(uiProgram , THICKNESS , _thickness);

        CHECK_GL_ERROR;

        Vector3 view_dir = _camera->get_look_at() - _camera->get_eye();
        view_dir = mat_v2w.get_transpose().transform(view_dir);
        view_dir.normalize();
        glProgramUniform3f(uiProgram , RAY_DIRECTION , (float)view_dir.x ,(float)view_dir.y , (float)view_dir.z);

        CHECK_GL_ERROR;

        const unsigned int aLocalWorkGroupCount[2] = {4,4};
        unsigned int aWorkGroupsNum[2] = {(unsigned int)_width/aLocalWorkGroupCount[0], (unsigned int)_height/aLocalWorkGroupCount[1]};
        if (aWorkGroupsNum[0]*aLocalWorkGroupCount[0] != (unsigned int)_width)
        {
            aWorkGroupsNum[0] +=1;
        }
        if (aWorkGroupsNum[1]*aLocalWorkGroupCount[1] != (unsigned int)_height)
        {
            aWorkGroupsNum[1] +=1;
        }
        glDispatchCompute(aWorkGroupsNum[0] , aWorkGroupsNum[1] , 1);

        CHECK_GL_ERROR;
        
        glPopAttrib();

        CHECK_GL_ERROR;

        _program->unbind();

        CHECK_GL_ERROR;

        //////////////////////////////////////////////////////////////////////////
        //For testing
        //std::cout << "Size : " << _width << " " << _height << std::endl;
        //_entry_points_texture->bind();
        //_entry_points_texture->download(GL_RGBA , GL_FLOAT , _entry_points_buffer.get());

        //_entry_points_texture->unbind();
        //_exit_points_texture->bind();
        //_exit_points_texture->download(GL_RGBA , GL_FLOAT , _exit_points_buffer.get());

        //this->debug_output_entry_points("D:/temp/gpu_MPR_ee.raw");

        CHECK_GL_ERROR;

#undef IMAGE_ENTRY_POINT
#undef IMAGE_EXIT_POINT
#undef DISPLAY_SIZE
#undef VOLUME_DIM
#undef MVP_INVERSE
#undef THICKNESS
#undef RAY_DIRECTION

    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::initialize()
{
    EntryExitPoints::initialize();

    if (GPU_BASE == _strategy)
    {
        if (!_program)
        {
            UIDType uid = 0;
            _program = GLResourceManagerContainer::instance()->get_program_manager()->create_object(uid);
            _program->set_description("MPR entry exit program");
            _program->set_shaders(std::vector<GLShaderInfo>(1 , GLShaderInfo(GL_COMPUTE_SHADER , S_MPR_ENTRY_EXIT_POINTS_COMP , "MPR entry exit compute shader")));
            _program->initialize();
            _program->compile();
        }
    }
}

void MPREntryExitPoints::finialize()
{
    EntryExitPoints::finialize();

    if (_program)
    {
        GLResourceManagerContainer::instance()->get_program_manager()->remove_object(_program->get_uid());
        _program.reset();
        GLResourceManagerContainer::instance()->get_program_manager()->update();
    }
}





MED_IMG_END_NAMESPACE