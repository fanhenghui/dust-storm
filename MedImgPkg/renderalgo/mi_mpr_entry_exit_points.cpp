#include "mi_mpr_entry_exit_points.h"

#include "boost/thread.hpp"
#include <limits>
#include <time.h>

#include "arithmetic/mi_arithmetic_utils.h"
#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_vector3f.h"
#include "io/mi_image_data.h"

#include "glresource/mi_gl_program.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_utils.h"

#include "mi_camera_calculator.h"
#include "mi_shader_collection.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
// Return true if out
bool check_outside(Vector3f pt, Vector3f bound) {
    const float BB_EPSILON = 1e-4f;
    if (pt._m[0] < -BB_EPSILON || pt._m[1] < -BB_EPSILON || pt._m[2] < -BB_EPSILON || 
        pt._m[0] > bound._m[0] + BB_EPSILON || pt._m[1] > bound._m[1] + BB_EPSILON || pt._m[2] > bound._m[2] + BB_EPSILON) {
        return true;
    } else {
        return false;
    }
}

// If ray[i] < FLOAT_EPSLION then set ray[i] = 1 adjust[i] =
// std::numeric_limits<float>::max()*0.5f
bool ray_intersect_aabb_acc(Vector3f ray_start, Vector3f min, Vector3f bound, Vector3f ray_norm, float& entry_step, float& exit_step) {
    
    Vector3f ray_r = Vector3f(1.0f,1.0f,1.0f) / ray_norm;
    Vector3f bottom_step = (min - ray_start);
    Vector3f top_step = (min + bound - ray_start);
    Vector3f bottom_step2 = bottom_step * ray_r;
    Vector3f top_step2  = top_step * ray_r;

    if(fabs(bottom_step.get_x()) < FLOAT_EPSILON){
        bottom_step2.set_x(0);
    }
    if(fabs(bottom_step.get_y()) < FLOAT_EPSILON){
        bottom_step2.set_y(0);
    }
    if(fabs(bottom_step.get_z()) < FLOAT_EPSILON){
        bottom_step2.set_z(0);
    }

    if(fabs(top_step.get_x()) < FLOAT_EPSILON){
        top_step2.set_x(0);
    }
    if(fabs(top_step.get_y()) < FLOAT_EPSILON){
        top_step2.set_y(0);
    }
    if(fabs(top_step.get_z()) < FLOAT_EPSILON){
        top_step2.set_z(0);
    }

    entry_step = bottom_step2.min_per_elem(top_step2).max_elem();
    exit_step = bottom_step2.max_per_elem(top_step2).min_elem();

    //////////////////////////////////////////////////////////////////////////
    // fNear > fFar not intersected
    // fNear >0  fFar > 0 fNear <= fFar intersected , start point not arrive AABB
    // yet
    // fNear <0 fFar > 0 intersected , start point is in AABB
    // fNear <0 fFar < 0 fNear < fFar , intersected , but start point is outside
    // AABB in extension ray
    return entry_step < exit_step;
}
}

MPREntryExitPoints::MPREntryExitPoints()
    : _thickness(1.0f), _sample_rate(1.0), 
    _entry_plane(1,0,0,0), _exit_plane(1,0,0,0),_ray_dir_norm(0,0,0),
    _standard_steps(0) {}

MPREntryExitPoints::~MPREntryExitPoints() {}

void MPREntryExitPoints::set_sample_rate(float sample_rate) {
    _sample_rate = sample_rate;
}

void MPREntryExitPoints::set_thickness(float thickness) {
    _thickness = thickness;
}

void MPREntryExitPoints::calculate_entry_exit_points() {
    MI_RENDERALGO_LOG(MI_TRACE) << "IN calculate MPR entry exit points.";
    _standard_steps = float(int(_thickness / _sample_rate + 0.5f));

    // clock_t t0 = clock();
    if (CPU_BASE == _strategy) {
        cal_entry_exit_points_cpu();
    } else if (GPU_BASE == _strategy) {
        cal_entry_exit_points_gpu();
    }
    // clock_t t1 = clock();
    // MI_RENDERALGO_LOG(MI_DEBUG) << "Calculate entry exit points cost : " << double(t1 - t0)/CLOCKS_PER_SEC;
    MI_RENDERALGO_LOG(MI_TRACE) << "OUT calculate MPR entry exit points.";
}

void MPREntryExitPoints::cal_entry_exit_points_cpu() {
    try {
        RENDERALGO_CHECK_NULL_EXCEPTION(_camera);
        RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume_data);

        const Vector3f dim_vector((float)_volume_data->_dim[0],
                                  (float)_volume_data->_dim[1],
                                  (float)_volume_data->_dim[2]);

        // Calculate base plane of MPR
        const Matrix4 mat_v2w = _camera_calculator->get_volume_to_world_matrix();
        const Matrix4 mat_vp = _camera->get_view_projection_matrix();
        const Matrix4 mat_mvp = mat_vp * mat_v2w;
        const Matrix4 mat_mvp_inv = mat_mvp.get_inverse();

        /*const Point3 pt00 = mat_mvp_inv.transform(Point3(-1.0,-1.0,0));
        const Point3 pt01 = mat_mvp_inv.transform(Point3(-1.0,1.0,0));
        const Point3 pt10 = mat_mvp_inv.transform(Point3(1.0,-1.0,0));*/

        Point2 pt00_2 =
            ArithmeticUtils::dc_to_ndc(Point2(0, _height - 1), _width, _height);
        Point2 pt01_2 = ArithmeticUtils::dc_to_ndc(Point2(0, 0), _width, _height);
        Point2 pt10_2 = ArithmeticUtils::dc_to_ndc(Point2(_width - 1, _height - 1),
                        _width, _height);
        const Point3 pt00 = mat_mvp_inv.transform(Point3(pt00_2.x, pt00_2.y, 0));
        const Point3 pt01 = mat_mvp_inv.transform(Point3(pt01_2.x, pt01_2.y, 0));
        const Point3 pt10 = mat_mvp_inv.transform(Point3(pt10_2.x, pt10_2.y, 0));

        const Vector3 x_delta = (pt10 - pt00) * (1.0 / (_width - 1));
        const Vector3 y_delta = (pt01 - pt00) * (1.0 / (_height - 1));

        Vector3 view_dir = _camera->get_look_at() - _camera->get_eye();
        view_dir = mat_v2w.get_transpose().transform(view_dir);
        view_dir.normalize();

        const Vector3f ray_dir = ArithmeticUtils::convert_vector(view_dir);

        const Vector3f pt00F((float)pt00.x, (float)pt00.y, (float)pt00.z);
        const Vector3f x_delta_float((float)x_delta.x, (float)x_delta.y,
                                     (float)x_delta.z);
        const Vector3f y_delta_float((float)y_delta.x, (float)y_delta.y,
                                     (float)y_delta.z);

        const float thickness = _thickness;
        const float thickness_half = thickness * 0.5f;
        Vector4f* entry_points_array = _entry_points_buffer.get();
        Vector4f* exit_points_array = _exit_points_buffer.get();

        //////////////////////////////////////////////////////////////////////////
        // Adjust ray direction
        Vector3f vRayBrick(ray_dir);
        for (int i = 0; i < 3; ++i) {
            if (fabs(vRayBrick._m[i]) <= FLOAT_EPSILON) {
                vRayBrick._m[i] = FLOAT_EPSILON; // be divided
            }
        }

        //////////////////////////////////////////////////////////////////////////

        const int pixel_sum = _width * _height;
#ifndef _DEBUG
        #pragma omp parallel for
#endif

        for (int idx = 0; idx < pixel_sum; ++idx) {
            Vector3f cur_f;
            Vector3f entry_f;
            Vector3f exit_f;
            Vector3f entry_intersection;
            Vector3f exit_intersection;

            int iY = idx / _width;
            int iX = idx - iY * _width;

            //if (idx == pixel_sum/2)
            //{
            //    printf(".");
            //}

            cur_f = pt00F + x_delta_float * (float)iX + y_delta_float * (float)iY;

            if (thickness <= 1.0) {
                entry_f = cur_f;
                exit_f = cur_f + ray_dir * thickness_half * 2;
            } else {
                entry_f = cur_f - ray_dir * thickness_half;
                exit_f = cur_f + ray_dir * thickness_half;
            }

            entry_intersection = entry_f;
            exit_intersection = exit_f;

            // Intersect volume AABB to get intersected entry&exit points
            float entry_step(0), exit_step(0);
            const bool bIntersection = ray_intersect_aabb_acc(
                                           entry_f, Vector3f(0, 0, 0), dim_vector, vRayBrick,
                                           entry_step, exit_step);

            // Entry point outside
            if (check_outside(entry_f, dim_vector)) {
                if (!bIntersection || entry_step < 0 ||
                        entry_step > thickness) // check entry points in range of thickness
                    // and volume
                {
                    entry_points_array[idx] = Vector4f(0, 0, 0, -1.0f);
                    exit_points_array[idx] = Vector4f(0, 0, 0, -1.0f);
                    continue;
                }

                entry_intersection = entry_f + ray_dir * entry_step;
            }

            // Exit point outside
            if (check_outside(exit_f, dim_vector)) {
                if (!bIntersection) {
                    entry_points_array[idx] = Vector4f(0, 0, 0, -1.0f);
                    exit_points_array[idx] = Vector4f(0, 0, 0, -1.0f);
                    continue;
                }

                exit_intersection = entry_f + ray_dir * exit_step;
                if (thickness <= 1.0) {
                    exit_intersection = entry_f + ray_dir * thickness;
                }
            }

            //////////////////////////////////////////////////////////////////////////
            // alpha value : ray step
            float fStep = (float)(int)((exit_intersection - entry_intersection).magnitude() / _sample_rate + 0.5f);
            if (fStep > _standard_steps) // Adjust step to prevent  fStep = standard
                // step + epsilon which it's ceil equals (
                // standard cell + 1)
            {
                fStep = _standard_steps;
            }

            entry_points_array[idx] = Vector4f(entry_intersection, 0.0f); // Entry step is 0 , the first sample
            // position is on entry plane
            exit_points_array[idx] = Vector4f(exit_intersection, fStep); // Exit step is integer step which
            // represent the integeration path

            //////////////////////////////////////////////////////////////////////////
        }

        //initialize();
        //_entry_points_texture->bind();
        //_entry_points_texture->load(GL_RGBA32F , _width , _height , GL_RGBA ,
        //    GL_FLOAT , _entry_points_buffer.get());

        //_exit_points_texture->bind();
        //_exit_points_texture->load(GL_RGBA32F , _width , _height , GL_RGBA ,
        //    GL_FLOAT , _exit_points_buffer.get());

        /*_entry_points_texture->bind();
        _entry_points_texture->download(GL_RGBA , GL_FLOAT ,
        _entry_points_buffer.get());

        _exit_points_texture->bind();
        _exit_points_texture->download(GL_RGBA , GL_FLOAT ,
        _exit_points_buffer.get());*/
    } catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << "calculate CPU MPR entry exit points failed with exception: " << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::get_entry_exit_plane(Vector4f& entry_point,
        Vector4f& exit_point,
        Vector3f& ray_dir_norm) {
    entry_point = _entry_plane;
    exit_point = _exit_plane;
    ray_dir_norm = _ray_dir_norm;
}

void MPREntryExitPoints::cal_entry_exit_points_gpu() {
    try {
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

        if (0 == uiProgram) {
            RENDERALGO_THROW_EXCEPTION("Program ID is 0!");
        }

        glPushAttrib(GL_ALL_ATTRIB_BITS);

        _program->bind();

        _entry_points_texture->bind_image(IMAGE_ENTRY_POINT, 0, false, 0,
                                          GL_READ_WRITE, GL_RGBA32F);
        _exit_points_texture->bind_image(IMAGE_EXIT_POINT, 0, false, 0,
                                         GL_READ_WRITE, GL_RGBA32F);

        glProgramUniform2ui(uiProgram, DISPLAY_SIZE, (GLuint)_width,
                            (GLuint)_height);

        const float fDim[3] = {(float)_volume_data->_dim[0],
                               (float)_volume_data->_dim[1],
                               (float)_volume_data->_dim[2]
                              };
        glProgramUniform3f(uiProgram, VOLUME_DIM, fDim[0], fDim[1], fDim[2]);

        const Matrix4 mat_v2w = _camera_calculator->get_volume_to_world_matrix();
        const Matrix4 mat_vp = _camera->get_view_projection_matrix();
        const Matrix4 mat_mvp = mat_vp * mat_v2w;
        const Matrix4 mat_mvp_inv = mat_mvp.get_inverse();

        float fMat[16] = {0.0f};
        mat_mvp_inv.to_float16(fMat);
        glProgramUniformMatrix4fv(uiProgram, MVP_INVERSE, 1, GL_FALSE, fMat);

        glProgramUniform1f(uiProgram, THICKNESS, _thickness);

        Vector3 view_dir = _camera->get_look_at() - _camera->get_eye();
        view_dir = mat_v2w.get_transpose().transform(view_dir);
        view_dir.normalize();
        glProgramUniform3f(uiProgram, RAY_DIRECTION, (float)view_dir.x,
                           (float)view_dir.y, (float)view_dir.z);

        const unsigned int aLocalWorkGroupCount[2] = {4, 4};
        unsigned int aWorkGroupsNum[2] = {
            (unsigned int)_width / aLocalWorkGroupCount[0],
            (unsigned int)_height / aLocalWorkGroupCount[1]
        };

        if (aWorkGroupsNum[0] * aLocalWorkGroupCount[0] != (unsigned int)_width) {
            aWorkGroupsNum[0] += 1;
        }

        if (aWorkGroupsNum[1] * aLocalWorkGroupCount[1] != (unsigned int)_height) {
            aWorkGroupsNum[1] += 1;
        }

        glDispatchCompute(aWorkGroupsNum[0], aWorkGroupsNum[1], 1);

        _program->unbind();

        glPopAttrib();

        CHECK_GL_ERROR;

        //////////////////////////////////////////////////////////////////////////
        // Test code
        /*MI_RENDERALGO_LOG(MI_DEBUG)  << "Size : " << _width << " " << _height << std::endl;
        _entry_points_texture->bind();
        _entry_points_texture->download(GL_RGBA , GL_FLOAT ,
            _entry_points_buffer.get());

        _entry_points_texture->unbind();
        _exit_points_texture->bind();
        _exit_points_texture->download(GL_RGBA , GL_FLOAT ,
            _exit_points_buffer.get());

        this->debug_output_entry_points("d:/temp/entry_points.raw");
        this->debug_output_exit_points("d:/temp/exit_points.raw");*/
        //////////////////////////////////////////////////////////////////////////

#undef IMAGE_ENTRY_POINT
#undef IMAGE_EXIT_POINT
#undef DISPLAY_SIZE
#undef VOLUME_DIM
#undef MVP_INVERSE
#undef THICKNESS
#undef RAY_DIRECTION

    } catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << "calculate GPU MPR entry exit points failed with exception: " << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::initialize() {
    EntryExitPoints::initialize();

    if (GPU_BASE == _strategy) {
        if (!_program) {
            UIDType uid = 0;
            _program = GLResourceManagerContainer::instance()
                       ->get_program_manager()
                       ->create_object(uid);
            _program->set_description("MPR entry exit program");
            _program->set_shaders(std::vector<GLShaderInfo>(
                                      1, GLShaderInfo(GL_COMPUTE_SHADER, S_MPR_ENTRY_EXIT_POINTS_COMP,
                                              "MPR entry exit compute shader")));
            _program->initialize();
            _program->compile();

            _res_shield.add_shield<GLProgram>(_program);
        }
    }
}

MED_IMG_END_NAMESPACE