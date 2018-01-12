#include "mi_ray_casting_gpu_cuda.h"
#include <cuda_runtime.h>

#include "arithmetic/mi_camera_base.h"
#include "arithmetic/mi_cuda_graphic.h"

#include "io/mi_image_data.h"

#include "cudaresource/mi_cuda_resource_manager.h"
#include "cudaresource/mi_cuda_gl_texture_2d.h"
#include "cudaresource/mi_cuda_texture_1d.h"
#include "cudaresource/mi_cuda_texture_1d_array.h"
#include "cudaresource/mi_cuda_surface_2d.h"
#include "cudaresource/mi_cuda_texture_3d.h"
#include "cudaresource/mi_cuda_global_memory.h"
#include "cudaresource/mi_cuda_utils.h"
#include "cudaresource/mi_cuda_time_query.h"

#include "glresource/mi_gl_texture_2d.h"

#include "mi_ray_caster.h"
#include "mi_entry_exit_points.h"
#include "mi_ray_caster_canvas.h"
#include "mi_camera_calculator.h"
#include "mi_ray_caster_inner_resource.h"

MED_IMG_BEGIN_NAMESPACE


//-------------------------------------------------//
//CUDA method
extern "C"
cudaError_t ray_cast_texture(cudaTextureObject_t entry_tex, cudaTextureObject_t exit_tex, int2 entry_exit_size,
    CudaVolumeInfos volume_info, CudaRayCastInfos ray_cast_info, cudaSurfaceObject_t canvas, int2 canvas_size, int quarter);

extern "C"
cudaError_t ray_cast_surface(cudaSurfaceObject_t entry_suf, cudaSurfaceObject_t exit_suf, int2 entry_exit_size,
    CudaVolumeInfos volume_info, CudaRayCastInfos ray_cast_info, cudaSurfaceObject_t canvas, int2 canvas_size, int quarter);

extern "C"
cudaError_t quarter_map_back(cudaSurfaceObject_t quarter_canvas, int2 quarter_size, cudaSurfaceObject_t canvas, int2 canvas_size);
//-------------------------------------------------//

struct RayCastingGPUCUDA::InnerResource {
    CudaGLTexture2DPtr entry_points_tex;
    CudaGLTexture2DPtr exit_points_tex;
    CudaSurface2DPtr quarter_canvas;
    CudaTimeQueryPtr time_query;

    InnerResource(){
        entry_points_tex = CudaResourceManager::instance()->create_cuda_gl_texture_2d(
            "cuda ray-cast inner cuda-gl texture for gl entry points");
        exit_points_tex = CudaResourceManager::instance()->create_cuda_gl_texture_2d(
            "cuda ray-cast inner cuda-gl texture for gl entry points");
        time_query = CudaResourceManager::instance()->create_cuda_time_query("cuda ray-cast inner time query");
    }
};

RayCastingGPUCUDA::RayCastingGPUCUDA(std::shared_ptr<RayCaster> ray_caster) : _ray_caster(ray_caster), _duration(0.0f){

}

RayCastingGPUCUDA::~RayCastingGPUCUDA() {

}

void RayCastingGPUCUDA::render() {

    std::shared_ptr<RayCaster> ray_caster = _ray_caster.lock();
    RENDERALGO_CHECK_NULL_EXCEPTION(ray_caster);

    std::shared_ptr<EntryExitPoints> entry_exit_points = ray_caster->get_entry_exit_points();
    std::shared_ptr<RayCasterCanvas> ray_caster_canvas = ray_caster->get_canvas();
    GPUCanvasPairPtr canvas_pair = ray_caster_canvas->get_color_attach_texture();
    RENDERALGO_CHECK_NULL_EXCEPTION(canvas_pair);

    CudaSurface2DPtr canvas_surface = canvas_pair->get_cuda_resource();
    RENDERALGO_CHECK_NULL_EXCEPTION(canvas_surface);

    CudaVolumeInfos cuda_volume_infos;
    CudaRayCastInfos cuda_ray_infos;
    fill_paramters(ray_caster, cuda_volume_infos, cuda_ray_infos);
    
    //TODO canvas downsample
    int width(0), height(0);
    entry_exit_points->get_display_size(width, height);

    const GPUPlatform gpu_platform = entry_exit_points->get_gpu_platform();
    if (CUDA_BASE == gpu_platform) {
        GPUCanvasPairPtr entry_pair = entry_exit_points->get_entry_points_texture();
        GPUCanvasPairPtr exit_pair = entry_exit_points->get_exit_points_texture();
        RENDERALGO_CHECK_NULL_EXCEPTION(entry_pair);
        RENDERALGO_CHECK_NULL_EXCEPTION(exit_pair);
        CudaSurface2DPtr entry = entry_pair->get_cuda_resource();
        CudaSurface2DPtr exit = exit_pair->get_cuda_resource();
        RENDERALGO_CHECK_NULL_EXCEPTION(entry);
        RENDERALGO_CHECK_NULL_EXCEPTION(exit);
        
        ScopedCudaTimeQuery inner_timer(_inner_resource->time_query, &_duration);
        cudaError_t err = cudaSuccess;
        const int2 entry_exit_points_size = make_int2(width, height);
        if (ray_caster->map_quarter_canvas()) {
            //-----------------------------------------------------//
            //downsample render to inner quarter canvas, then map to r-c canvas
            const int quarter_width = width >> 1;
            const int quarter_height = height >> 1;
            if (nullptr == _inner_resource->quarter_canvas) {
                _inner_resource->quarter_canvas = CudaResourceManager::instance()->
                    create_cuda_surface_2d("cuda r-c inner quarter canvas for downsample");
            }
            //reload size
            if (0 != _inner_resource->quarter_canvas->load(8, 8, 8, 8, cudaChannelFormatKindUnsigned, quarter_width, quarter_height, nullptr)) {
                RENDERALGO_THROW_EXCEPTION("cuda r-c inner quarter canvas malloc failed.");
            }
            const int2 canvas_size = make_int2(quarter_width, quarter_height);
            err = ray_cast_surface(entry->get_object(), exit->get_object(), entry_exit_points_size, 
                cuda_volume_infos, cuda_ray_infos, _inner_resource->quarter_canvas->get_object(), canvas_size, 1);
            if (err != cudaSuccess) {
                LOG_CUDA_ERROR(err);
                RENDERALGO_THROW_EXCEPTION("cuda rc-surface in quarter failed.");
            }

            err = quarter_map_back(_inner_resource->quarter_canvas->get_object(), canvas_size, canvas_surface->get_object(), entry_exit_points_size);
            if (err != cudaSuccess) {
                LOG_CUDA_ERROR(err);
                RENDERALGO_THROW_EXCEPTION("cuda rc-surface quarter map back failed.");
            }

        } else {
            err = ray_cast_surface(entry->get_object(), exit->get_object(), entry_exit_points_size, 
                cuda_volume_infos, cuda_ray_infos, canvas_surface->get_object(), entry_exit_points_size, 0);
            if (err != cudaSuccess) {
                LOG_CUDA_ERROR(err);
                RENDERALGO_THROW_EXCEPTION("cuda rc-surface failed.");
            }
        }
        
    } else {

        GPUCanvasPairPtr entry_pair = entry_exit_points->get_entry_points_texture();
        GPUCanvasPairPtr exit_pair = entry_exit_points->get_exit_points_texture();
        RENDERALGO_CHECK_NULL_EXCEPTION(entry_pair);
        RENDERALGO_CHECK_NULL_EXCEPTION(exit_pair);
        GLTexture2DPtr entry = entry_pair->get_gl_resource();
        GLTexture2DPtr exit = exit_pair->get_gl_resource();
        RENDERALGO_CHECK_NULL_EXCEPTION(entry);
        RENDERALGO_CHECK_NULL_EXCEPTION(exit);

        //register entry exit points GL texture
        if (nullptr == _inner_resource) {
            _inner_resource.reset(new InnerResource());
            if (0 != _inner_resource->entry_points_tex->register_gl_texture(entry, cudaGraphicsRegisterFlagsReadOnly)) {
                RENDERALGO_THROW_EXCEPTION("register entry points failed.");
            }
            if (0 != _inner_resource->exit_points_tex->register_gl_texture(exit, cudaGraphicsRegisterFlagsReadOnly)) {
                RENDERALGO_THROW_EXCEPTION("register exit points failed.");
            }
        } else {
            if (!_inner_resource->entry_points_tex->valid()) {
                if (0 != _inner_resource->entry_points_tex->register_gl_texture(entry, cudaGraphicsRegisterFlagsReadOnly)) {
                    RENDERALGO_THROW_EXCEPTION("register entry points failed.");
                }
            }
            if (!_inner_resource->exit_points_tex->valid()) {
                if (0 != _inner_resource->exit_points_tex->register_gl_texture(exit, cudaGraphicsRegisterFlagsReadOnly)) {
                    RENDERALGO_THROW_EXCEPTION("register entry points failed.");
                }
            }
        }
           
        if (0 != _inner_resource->entry_points_tex->map_gl_texture()) {
            RENDERALGO_THROW_EXCEPTION("map entry points failed.");
        }
        if (0 != _inner_resource->exit_points_tex->map_gl_texture()) {
            RENDERALGO_THROW_EXCEPTION("map exit points failed.");
        }

        cudaTextureObject_t entry_cuda_tex = _inner_resource->entry_points_tex->
            get_object(cudaAddressModeClamp, cudaFilterModePoint, cudaReadModeElementType, false);

        cudaTextureObject_t exit_cuda_tex = _inner_resource->exit_points_tex->
            get_object(cudaAddressModeClamp, cudaFilterModePoint, cudaReadModeElementType, false);
        

        ScopedCudaTimeQuery inner_timer(_inner_resource->time_query, &_duration);
        cudaError_t err = cudaSuccess;
        const int2 entry_exit_points_size = make_int2(width, height);
        if (ray_caster->map_quarter_canvas()) {
            //-----------------------------------------------------//
            //downsample render to inner quarter canvas, then map to r-c canvas
            const int quarter_width = width >> 1;
            const int quarter_height = height >> 1;
            if (nullptr == _inner_resource->quarter_canvas) {
                _inner_resource->quarter_canvas = CudaResourceManager::instance()->
                    create_cuda_surface_2d("cuda r-c inner quarter canvas for downsample");
            }
            //reload size
            if (0 != _inner_resource->quarter_canvas->load(8, 8, 8, 8, cudaChannelFormatKindUnsigned, quarter_width, quarter_height, nullptr)) {
                RENDERALGO_THROW_EXCEPTION("cuda r-c inner quarter canvas malloc failed.");
            }
            const int2 canvas_size = make_int2(quarter_width, quarter_height);
            err = ray_cast_texture(entry_cuda_tex, exit_cuda_tex, entry_exit_points_size, 
                cuda_volume_infos, cuda_ray_infos, _inner_resource->quarter_canvas->get_object(), canvas_size, 1);

            if (err != cudaSuccess) {
                LOG_CUDA_ERROR(err);
                _inner_resource->entry_points_tex->unmap_gl_texture();
                _inner_resource->exit_points_tex->unmap_gl_texture();
                RENDERALGO_THROW_EXCEPTION("cuda rc-surface in quarter failed.");
            }
            if (0 != _inner_resource->entry_points_tex->unmap_gl_texture()) {
                RENDERALGO_THROW_EXCEPTION("unmap entry points failed.");
            }
            if (0 != _inner_resource->exit_points_tex->unmap_gl_texture()) {
                RENDERALGO_THROW_EXCEPTION("unmap exit points failed.");
            }

            err = quarter_map_back(_inner_resource->quarter_canvas->get_object(), canvas_size, canvas_surface->get_object(), entry_exit_points_size);
            if (err != cudaSuccess) {
                LOG_CUDA_ERROR(err);
                RENDERALGO_THROW_EXCEPTION("cuda rc-surface quarter map back failed.");
            }
        }
        else {
            err = ray_cast_texture(entry_cuda_tex, exit_cuda_tex, entry_exit_points_size, 
                cuda_volume_infos, cuda_ray_infos, canvas_surface->get_object(), entry_exit_points_size , 0);
            if (err != cudaSuccess) {
                LOG_CUDA_ERROR(err);
                _inner_resource->entry_points_tex->unmap_gl_texture();
                _inner_resource->exit_points_tex->unmap_gl_texture();
                RENDERALGO_THROW_EXCEPTION("cuda rc-surface in quarter failed.");
            }
            if (0 != _inner_resource->entry_points_tex->unmap_gl_texture()) {
                RENDERALGO_THROW_EXCEPTION("unmap entry points failed.");
            }
            if (0 != _inner_resource->exit_points_tex->unmap_gl_texture()) {
                RENDERALGO_THROW_EXCEPTION("unmap exit points failed.");
            }
        }
    }
}

void RayCastingGPUCUDA::on_entry_exit_points_resize(int width, int height) {
    if (_inner_resource) {
        _inner_resource->entry_points_tex->unregister_gl_texture();
        _inner_resource->exit_points_tex->unregister_gl_texture();
    }
}

void RayCastingGPUCUDA::fill_paramters(std::shared_ptr<RayCaster> ray_caster,
    CudaVolumeInfos& cuda_volume_infos, CudaRayCastInfos& cuda_ray_infos) {
    //--------------------------------//
    //volume infos
    //--------------------------------//
    std::shared_ptr<ImageData> volume = ray_caster->get_volume_data();
    cuda_volume_infos.dim = make_uint3(volume->_dim[0], volume->_dim[1], volume->_dim[2]);
    cuda_volume_infos.dim_r = make_float3(1.0f/(float)volume->_dim[0], 1.0f / (float)volume->_dim[1], 1.0f / (float)volume->_dim[2]);
    cuda_volume_infos.sample_shift = 0.5f *  cuda_volume_infos.dim_r * 
        make_float3((float)volume->_spacing[0], (float)volume->_spacing[1], (float)volume->_spacing[2]);

    GPUTexture3DPairPtr volume_tex = ray_caster->get_volume_data_texture();    
    RENDERALGO_CHECK_NULL_EXCEPTION(volume_tex);
    CudaTexture3DPtr cuda_volume_tex = volume_tex->get_cuda_resource();
    RENDERALGO_CHECK_NULL_EXCEPTION(cuda_volume_tex);
    cuda_volume_infos.volume_tex = cuda_volume_tex->get_object(cudaAddressModeClamp, 
        cudaFilterModeLinear, cudaReadModeNormalizedFloat, true);

    const MaskMode mask_mode = ray_caster->get_mask_mode();
    GPUTexture3DPairPtr mask_tex = ray_caster->get_mask_data_texture();
    if (MASK_NONE != mask_mode) {
        RENDERALGO_CHECK_NULL_EXCEPTION(mask_tex);
        CudaTexture3DPtr cuda_mask_tex = mask_tex->get_cuda_resource();
        RENDERALGO_CHECK_NULL_EXCEPTION(cuda_mask_tex);
        cuda_volume_infos.mask_tex = cuda_mask_tex->get_object(cudaAddressModeClamp, 
            cudaFilterModePoint, cudaReadModeNormalizedFloat, true);
    }

    //--------------------------------//
    //ray cast infos
    //--------------------------------//
    std::shared_ptr<CameraBase> camera = ray_caster->get_camera();

    //label level to restrict max rc-mask label number .
    cuda_ray_infos.label_level = (int)ray_caster->get_mask_label_level();

    //ray-casting mode
    cuda_ray_infos.mask_mode = (int)mask_mode;
    cuda_ray_infos.composite_mode = (int)ray_caster->get_composite_mode();
    cuda_ray_infos.interpolation_mode = (int)ray_caster->get_interpolation_mode();
    cuda_ray_infos.shading_mode = (int)ray_caster->get_shading_mode();
    cuda_ray_infos.color_inverse_mode = (int)ray_caster->get_color_inverse_mode();
    cuda_ray_infos.mask_overlay_mode = (int)ray_caster->get_mask_overlay_mode();

    //sample step
    cuda_ray_infos.sample_step = ray_caster->get_sample_step();

    //normal matrix
    std::shared_ptr<CameraCalculator> camera_cal = ray_caster->get_camera_calculator();
    const Matrix4 mat_v2w = camera_cal->get_volume_to_world_matrix();
    const Matrix4 mat_view = camera->get_view_matrix();
    const Matrix4 mat_normal = (mat_view * mat_v2w).get_inverse().get_transpose();
    cuda_ray_infos.mat_normal = matrix4_to_mat4(mat_normal);

    //calculate light position
    Point3 eye = camera->get_eye();
    Point3 lookat = camera->get_look_at();
    Vector3 view = camera->get_view_direction();
    const static double max_dim =
        (std::max)((std::max)(volume->_dim[0] * volume->_spacing[0],
            volume->_dim[1] * volume->_spacing[1]),
            volume->_dim[2] * volume->_spacing[2]);
    const static float magic_num = 1.5f;//distance
    Point3 light_pos = lookat - view * max_dim * magic_num;
    cuda_ray_infos.light_position = make_float3((float)light_pos.x, (float)light_pos.y, (float)light_pos.z);

    //illumination parameters: ambient attribute    
    float ambient[4];
    ray_caster->get_ambient_color(ambient);
    cuda_ray_infos.ambient_color = make_float3(ambient[0], ambient[1], ambient[2]);
    cuda_ray_infos.ambient_intensity = ambient[3];

    CudaGlobalMemoryPtr shared_mapped_memory = ray_caster->get_inner_resource()->get_shared_map_memory();
    cuda_ray_infos.d_shared_mapped_memory = shared_mapped_memory->get_pointer();

    //color opacity texture shift
    cuda_ray_infos.color_opacity_texture_shift = 0.5f / S_TRANSFER_FUNC_WIDTH;

    //global window
    float gww, gwl;
    ray_caster->get_global_window_level(gww, gwl);
    volume->normalize_wl(gww, gwl);
    cuda_ray_infos.global_ww = gww;
    cuda_ray_infos.global_wl = gwl;

    //pseudo color texture 
    unsigned int lut_length = S_TRANSFER_FUNC_WIDTH;
    GPUTexture1DPairPtr pseudo_texture = ray_caster->get_pseudo_color_texture(lut_length);
    RENDERALGO_CHECK_NULL_EXCEPTION(pseudo_texture);
    CudaTexture1DPtr cuda_pseudo_texture = pseudo_texture->get_cuda_resource();
    RENDERALGO_CHECK_NULL_EXCEPTION(cuda_pseudo_texture);
    cuda_ray_infos.pseudo_color_texture = cuda_pseudo_texture->get_object(
        cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeNormalizedFloat, true);
    cuda_ray_infos.pseudo_color_texture_shift = 0.5f / lut_length;

    //test code
    cuda_ray_infos.test_code = ray_caster->get_test_code();
}

float RayCastingGPUCUDA::get_rendering_duration() const {
    return _duration;
}

MED_IMG_END_NAMESPACE