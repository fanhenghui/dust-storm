#include "mi_graphic_object_navigator.h"

#include <cuda_runtime.h>

#include "util/mi_file_util.h"
#include "util/mi_memory_shield.h"

#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_cuda_graphic.h"

#include "glresource/mi_gl_utils.h"
#include "glresource/mi_gl_texture_2d.h"
#include "glresource/mi_gl_texture_cache.h"

#include "cudaresource/mi_cuda_surface_2d.h"
#include "cudaresource/mi_cuda_texture_2d.h"
#include "cudaresource/mi_cuda_global_memory.h"
#include "cudaresource/mi_cuda_resource_manager.h"
#include "cudaresource/mi_cuda_utils.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

GraphicObjectNavigator::GraphicObjectNavigator(GPUPlatform platform) : _gpu_platform(platform), _has_init(false) {
    _width = 100;
    _height = 100;
    _x = 512 - _width - 20;
    _y = 512 - _height - 20;
}

GraphicObjectNavigator::~GraphicObjectNavigator() {

}

void GraphicObjectNavigator::initialize() {
    if(!_has_init) {
#ifdef WIN32
        const std::string navi_img_file("./config/resource/navi_384_256_3.raw");
#else
        const std::string navi_img_file("../config/resource/navi_384_256_3.raw");
#endif
        const unsigned int img_width = 384;
        const unsigned int img_height = 256;
        const unsigned int img_size = 384*256*3;
        unsigned char* rgb_array = new unsigned char[img_size];
        if( 0 != FileUtil::read_raw(navi_img_file, rgb_array, img_size) ) {
            MI_RENDERALGO_LOG(MI_FATAL) << "load navigator image failed.";
        } else {
            if (GL_BASE == _gpu_platform) {
                GLTexture2DPtr navi_tex = GLResourceManagerContainer::instance()->get_texture_2d_manager()->create_object("navigator map");
                _res_shield.add_shield<GLTexture2D>(navi_tex);
                _navi_tex.reset(new GPUTexture2DPair(navi_tex));
                GLTextureCache::instance()->cache_load(GL_TEXTURE_2D, navi_tex, GL_CLAMP_TO_BORDER,
                    GL_LINEAR, GL_RGB8, img_width, img_height, 1, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
            } else {
                unsigned char* rgba_array = new unsigned char[img_width * img_height * 4];
                MemShield shield(rgb_array);
                MemShield shield2(rgba_array);
                for (unsigned int i = 0; i < img_width * img_height; ++i) {
                    rgba_array[i * 4] = rgb_array[i * 3];
                    rgba_array[i * 4 + 1] = rgb_array[i * 3 + 1];
                    rgba_array[i * 4 + 2] = rgb_array[i * 3 + 2];
                    rgba_array[i * 4 + 3] = 255;
                }

                CudaTexture2DPtr navi_tex = CudaResourceManager::instance()->create_cuda_texture_2d("navigator map");
                _navi_tex.reset(new GPUTexture2DPair(navi_tex));
                if (0 != navi_tex->load(8, 8, 8, 8, cudaChannelFormatKindUnsigned, img_width, img_height, rgba_array)) {
                    MI_RENDERALGO_LOG(MI_FATAL) << "load navigator cuda texture failed.";
                }
            }
        }
        _has_init = true;
    }
}

void GraphicObjectNavigator::set_navi_position(int x, int y, int width, int height) {
    _x = x;
    _y = y;
    _width = width;
    _height = height;
}

void GraphicObjectNavigator::render(int code) {
    if (_navi_tex == nullptr) {
        return;
    }
    if (GL_BASE != _gpu_platform) {
        RENDERALGO_THROW_EXCEPTION("invalid GPU platform when render navigator to cuda surface.");
        return;
    }
    if (!_navi_tex->gl()) {
        RENDERALGO_THROW_EXCEPTION("invalid GPU platform");
    }
    RENDERALGO_CHECK_NULL_EXCEPTION(_navi_tex->get_gl_resource());

    OrthoCamera camera;
    if (_camera) {   
        Vector3 view = _camera->get_view_direction();
        Vector3 up = _camera->get_up_direction();
        camera.set_look_at(Point3::S_ZERO_POINT);
        Point3 eye = Point3::S_ZERO_POINT - view*3;
        camera.set_eye(eye);
        camera.set_up_direction(up);
        camera.set_ortho(-1,1,-1,1,1,5);
    }

    CHECK_GL_ERROR;
    glViewport(_x, _y, _width, _height);
    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(camera.get_view_projection_matrix()._m);

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);

    const float w = 0.6f;
    glEnable(GL_TEXTURE_2D);
    _navi_tex->get_gl_resource()->bind();
    glBegin(GL_QUADS);

    const float x_step = 0.33333f;
    const float y_step = 0.5f;
    //patient coordinate
    //head 
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(-w, -w, w);
    glTexCoord2f(x_step*3.0f, y_step);
    glVertex3f(w, -w, w);
    glTexCoord2f(x_step*3.0f, 0);
    glVertex3f(w, w, w);
    glTexCoord2f(x_step*2.0f, 0);
    glVertex3f(-w, w, w);

    //foot
    glTexCoord2f(x_step*2.0f, y_step*2.0f);
    glVertex3f(-w, -w, -w);
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(-w, w, -w);
    glTexCoord2f(x_step*3.0f, y_step);
    glVertex3f(w, w, -w);
    glTexCoord2f(x_step*3.0f, y_step*2.0f);
    glVertex3f(w, -w, -w);
    
    //left
    glTexCoord2f(x_step, 0);
    glVertex3f(-w, -w, -w);
    glTexCoord2f(x_step, y_step);
    glVertex3f(-w, -w, w);
    glTexCoord2f(0, y_step);
    glVertex3f(-w, w, w);
    glTexCoord2f(0, 0);
    glVertex3f(-w, w, -w);

    //right
    glTexCoord2f(0, y_step);
    glVertex3f(w, -w, -w);
    glTexCoord2f(x_step, y_step);
    glVertex3f(w, w, -w);
    glTexCoord2f(x_step, y_step*2);
    glVertex3f(w, w, w);
    glTexCoord2f(0, y_step*2);
    glVertex3f(w, -w, w);

    //posterior
    glTexCoord2f(x_step*2.0f, 0);
    glVertex3f(-w, w, -w);
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(-w, w, w);
    glTexCoord2f(x_step, y_step);
    glVertex3f(w, w, w);
    glTexCoord2f(x_step, 0);
    glVertex3f(w, w, -w);

    //anterior
    glTexCoord2f(x_step, y_step);
    glVertex3f(-w, -w, -w);
    glTexCoord2f(x_step*2.0f, y_step);
    glVertex3f(w, -w, -w);
    glTexCoord2f(x_step*2.0f, y_step*2.0f);
    glVertex3f(w, -w, w);
    glTexCoord2f(x_step, y_step*2.0f);
    glVertex3f(-w, -w, w);
    
    glEnd();  

    _navi_tex->get_gl_resource()->unbind();
    glPopMatrix();
    glPopAttrib();

    CHECK_GL_ERROR;
}

extern "C"
cudaError_t ray_tracing_quad_vertex_mapping(Viewport viewport, int width, int height,
    mat4 mat_viewmodel, mat4 mat_projection_inv, mat4 mat_mvp,
    int vertex_count, float3* d_vertex, float2* d_tex_coordinate, cudaTextureObject_t mapping_tex, cudaSurfaceObject_t canvas_surf, bool is_blend);

struct  GraphicObjectNavigator::InnerCudaResource {
    CudaGlobalMemoryPtr d_vertex_array;
    CudaGlobalMemoryPtr d_tex_coordinate_array;

    InnerCudaResource() {
        const float w = 0.6f;
        const float x_step = 0.33333f;
        const float y_step = 0.5f;
        //---------------------------------//
        //triangle
        float vertex[] = {
            -w, -w, w,
            w, -w, w,
            w, w, w,
            -w, w, w,
            -w, -w, -w,
            -w, w, -w,
            w, w, -w,
            w, -w, -w,
            -w, -w, -w,
            -w, -w, w,
            -w, w, w,
            -w, w, -w,
            w, -w, -w,
            w, w, -w,
            w, w, w,
            w, -w, w,
            -w, w, -w,
            -w, w, w,
            w, w, w,
            w, w, -w,
            -w, -w, -w,
            w, -w, -w,
            w, -w, w,
            -w, -w, w
        };

        float tex_coordinate[] = {
            x_step*2.0f, y_step,
            x_step*3.0f, y_step,
            x_step*3.0f, 0,
            x_step*2.0f, 0,
            x_step*2.0f, y_step*2.0f,
            x_step*2.0f, y_step,
            x_step*3.0f, y_step,
            x_step*3.0f, y_step*2.0f,
            x_step, 0,
            x_step, y_step,
            0, y_step,
            0, 0,
            0, y_step,
            x_step, y_step,
            x_step, y_step * 2,
            0, y_step * 2,
            x_step*2.0f, 0,
            x_step*2.0f, y_step,
            x_step, y_step,
            x_step, 0,
            x_step, y_step,
            x_step*2.0f, y_step,
            x_step*2.0f, y_step*2.0f,
            x_step, y_step*2.0f
        };

        d_vertex_array = CudaResourceManager::instance()->create_global_memory("navigator vertex array");
        d_tex_coordinate_array = CudaResourceManager::instance()->create_global_memory("navigator texture coordinate array");

        d_vertex_array->load(sizeof(vertex), vertex);
        d_tex_coordinate_array->load(sizeof(tex_coordinate), tex_coordinate);
    }

    ~InnerCudaResource() {

    }
};

void GraphicObjectNavigator::render_to_cuda_surface(CudaSurface2DPtr surface) {
    if (_navi_tex == nullptr) {
        return;
    }
    if (CUDA_BASE != _gpu_platform) {
        RENDERALGO_THROW_EXCEPTION("invalid GPU platform when render navigator to cuda surface.");
        return;
    }
    if (!_navi_tex->cuda()) {
        RENDERALGO_THROW_EXCEPTION("invalid GPU platform");
    }
    RENDERALGO_CHECK_NULL_EXCEPTION(_navi_tex->get_cuda_resource());

    if (nullptr == _inner_cuda_res) {
        _inner_cuda_res.reset(new InnerCudaResource());
    }
    
    OrthoCamera camera;
    if (_camera) {
        Vector3 view = _camera->get_view_direction();
        Vector3 up = _camera->get_up_direction();
        camera.set_look_at(Point3::S_ZERO_POINT);
        Point3 eye = Point3::S_ZERO_POINT - view * 3;
        camera.set_eye(eye);
        camera.set_up_direction(up);
        camera.set_ortho(-1, 1, -1, 1, 1, 5);
    }

    Matrix4 mat_v = camera.get_view_matrix();
    Matrix4 mat_p = camera.get_projection_matrix();
    Matrix4 mat_mvp = mat_p*mat_v;
    Matrix4 mat_pi = mat_p.get_inverse();
    mat4 mat4_v = matrix4_to_mat4(mat_v);
    mat4 mat4_pi = matrix4_to_mat4(mat_pi);
    mat4 mat4_mvp = matrix4_to_mat4(mat_mvp);

    const Viewport view_port(_x, _y, _width, _height);
    const int canvas_width = surface->get_width();
    const int canvas_height = surface->get_height();
    cudaTextureObject_t navi_tex = _navi_tex->get_cuda_resource()->get_object(cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeNormalizedFloat, true);

    cudaError_t err = ray_tracing_quad_vertex_mapping(view_port, canvas_width, canvas_height, mat4_v, mat4_pi, mat4_mvp, 24, 
        (float3*)_inner_cuda_res->d_vertex_array->get_pointer(), (float2*)_inner_cuda_res->d_tex_coordinate_array->get_pointer(),
        navi_tex , surface->get_object(), true);

    if (err != cudaSuccess) { 
        LOG_CUDA_ERROR(err);
        RENDERALGO_THROW_EXCEPTION("cuda render navigator failed.");
    }
}

MED_IMG_END_NAMESPACE