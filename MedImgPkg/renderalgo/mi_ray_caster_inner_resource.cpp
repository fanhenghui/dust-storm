#include "mi_ray_caster_inner_resource.h"
#include "glresource/mi_gl_buffer.h"
#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_utils.h"

#include "cudaresource/mi_cuda_global_memory.h"
#include "cudaresource/mi_cuda_resource_manager.h"
#include "cudaresource/mi_cuda_texture_1d_array.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE
struct RayCasterInnerResource::GLResource {
    std::map<RayCasterInnerResource::BufferType, GLBufferPtr> buffer_ids;
    bool dirty_flag[TYPE_END];
    GLResourceShield res_shield;

    GLResource() {
        memset(dirty_flag, 1, sizeof(bool) * TYPE_END);
    }

    void release() {
        buffer_ids.clear();
        memset(dirty_flag, 0, sizeof(bool) * TYPE_END);
    }

    std::string get_buffer_type_name(RayCasterInnerResource::BufferType type) {
        switch (type) {
        case WINDOW_LEVEL_BUCKET: {
            return "window level bucket";
        }

        case VISIBLE_LABEL_BUCKET: {
            return "visible label bucket";
        }

        case VISIBLE_LABEL_ARRAY: {
            return "visible label array";
        }

        case MASK_OVERLAY_COLOR_BUCKET: {
            return "mask overlay color bucket";
        }

        case MATERIAL_BUCKET: {
            return "material bucket";
        }

        default: {
            return "undefined buffer type";
        }
        }
    }

    GLBufferPtr GetBuffer(RayCasterInnerResource::BufferType type) {
        auto it = buffer_ids.find(type);

        if (it != buffer_ids.end()) {
            return it->second;
        } else {
            GLBufferPtr buffer = GLResourceManagerContainer::instance()->get_buffer_manager()
                ->create_object("ray caster inner buffer : " + get_buffer_type_name(type));
            buffer->initialize();
            buffer->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
            buffer_ids[type] = buffer;
            res_shield.add_shield<GLBuffer>(buffer);
            return buffer;
        }
    }
};

struct RayCasterInnerResource::CudaResource  {
    CudaGlobalMemoryPtr shared_map_memory;
    bool dirty_flag[TYPE_END];//Just use WINDOW_LEVEL_BUCKET, VISIBLE_LABEL_BUCKET, MATERIAL_BUCKET, MASK_OVERLAY_COLOR_BUCKET
    CudaResource() {
        memset(dirty_flag, 1, sizeof(bool) * TYPE_END);
    }
};

RayCasterInnerResource::RayCasterInnerResource(GPUPlatform gpu_platform)
    : _gpu_platform(gpu_platform), _label_level(L_8) {
    if (GL_BASE == _gpu_platform) {
        //---------------------------------------------//
        //GL set wl/material/visible to each gl-buffer
        _inner_gl_resource.reset(new GLResource());
        _shared_buffer_array.reset(new char[(int)_label_level * sizeof(Material)]);
    } else {
        //---------------------------------------------//
        //CUDA set all data to global memory, then copy to shared memory
        _inner_cuda_resource.reset(new CudaResource());
        _shared_buffer_array.reset(new char[(int)_label_level * 56]);//note 56 it the same with kernel
    }
}

RayCasterInnerResource::~RayCasterInnerResource() {}

GLBufferPtr RayCasterInnerResource::get_buffer(BufferType type) {
    try {
        if (GL_BASE != _gpu_platform) {
            RENDERALGO_THROW_EXCEPTION("invalid gpu platform.");
        }

        GLBufferPtr buffer = _inner_gl_resource->GetBuffer(type);
        CHECK_GL_ERROR;

        switch (type) {
        case WINDOW_LEVEL_BUCKET: {
            if (check_dirty(type)) {
                float* wl_array = (float*)_shared_buffer_array.get();
                memset(wl_array, 0, sizeof(float) * static_cast<int>(_label_level) * 2);

                for (auto it = _window_levels.begin(); it != _window_levels.end();
                        ++it) {
                    const unsigned char label = it->first;

                    if (label > static_cast<int>(_label_level) - 1) {
                        std::stringstream ss;
                        ss << "Input window level label : " << (int)(label)
                           << " is greater than the limit : "
                           << static_cast<int>(_label_level) - 1 << " !";
                        RENDERALGO_THROW_EXCEPTION(ss.str());
                    }

                    wl_array[label * 2] = it->second._value.x;
                    wl_array[label * 2 + 1] = it->second._value.y;
                }

                buffer->bind();
                buffer->load(static_cast<int>(_label_level) * sizeof(float) * 2,
                             wl_array, GL_STATIC_DRAW);
                
                remove_dirty(type);
            }

            break;
        }

        case VISIBLE_LABEL_BUCKET: {
            if (check_dirty(type)) {
                int* label_array = (int*)_shared_buffer_array.get();
                memset(label_array, 0, sizeof(int) * static_cast<int>(_label_level));

                for (auto it = _labels.begin(); it != _labels.end(); ++it) {
                    if (*it > static_cast<int>(_label_level) - 1) {
                        std::stringstream ss;
                        ss << "Input visible label : " << (int)(*it)
                           << " is greater than the limit : "
                           << static_cast<int>(_label_level) - 1 << " !";
                        RENDERALGO_THROW_EXCEPTION(ss.str());
                    }

                    label_array[*it] = 1;
                }

                buffer->bind();
                buffer->load(static_cast<int>(_label_level) * sizeof(int), label_array,
                             GL_STATIC_DRAW);

                remove_dirty(type);
            }

            break;
        }

        case VISIBLE_LABEL_ARRAY: {
            if (check_dirty(type)) {
                int* label_array = (int*)_shared_buffer_array.get();
                memset(label_array, 0, sizeof(int) * static_cast<int>(_label_level));

                int idx = 0;

                for (auto it = _labels.begin(); it != _labels.end(); ++it, ++idx) {
                    if (*it > static_cast<int>(_label_level) - 1) {
                        std::stringstream ss;
                        ss << "Input visible label : " << (int)(*it)
                           << " is greater than the limit : "
                           << static_cast<int>(_label_level) - 1 << " !";
                        RENDERALGO_THROW_EXCEPTION(ss.str());
                    }

                    label_array[idx] = static_cast<int>(*it);
                }

                buffer->bind();
                buffer->load(idx * sizeof(int), label_array, GL_STATIC_DRAW);

                remove_dirty(type);
            }

            break;
        }

        case MASK_OVERLAY_COLOR_BUCKET: {
            if (check_dirty(type)) {
                float* color_array = (float*)_shared_buffer_array.get();
                memset(color_array, 0,
                       sizeof(RGBAUnit) * static_cast<int>(_label_level));

                unsigned char label = 0;

                for (auto it = _mask_overlay_colors.begin();
                        it != _mask_overlay_colors.end(); ++it) {
                    label = it->first;

                    if (label > static_cast<int>(_label_level) - 1) {
                        std::stringstream ss;
                        ss << "Input visible label : " << (int)(it->first)
                           << " is greater than the limit : "
                           << static_cast<int>(_label_level) - 1 << " !";
                        RENDERALGO_THROW_EXCEPTION(ss.str());
                    }

                    color_array[label * 4] = it->second.r / 255.0f;
                    color_array[label * 4 + 1] = it->second.g / 255.0f;
                    color_array[label * 4 + 2] = it->second.b / 255.0f;
                    color_array[label * 4 + 3] = it->second.a / 255.0f;
                }

                buffer->bind();
                buffer->load(static_cast<int>(_label_level) * sizeof(float) * 4,
                             color_array, GL_STATIC_DRAW);

                remove_dirty(type);
            }

            break;
        }

        case MATERIAL_BUCKET: {
            if (check_dirty(type)) {
                Material* material_array = (Material*)_shared_buffer_array.get();
                memset(material_array, 0,
                       sizeof(Material) * static_cast<int>(_label_level));

                unsigned char label = 0;

                for (auto it = _material.begin(); it != _material.end(); ++it) {
                    label = it->first;

                    if (label > static_cast<int>(_label_level) - 1) {
                        std::stringstream ss;
                        ss << "Input visible label : " << (int)(it->first)
                           << " is greater than the limit : "
                           << static_cast<int>(_label_level) - 1 << " !";
                        RENDERALGO_THROW_EXCEPTION(ss.str());
                    }

                    material_array[label] = it->second;
                }

                buffer->bind();
                buffer->load(static_cast<int>(_label_level) * sizeof(Material),
                             material_array, GL_STATIC_DRAW);

                remove_dirty(type);
            }

            break;
        }

        default: {
            RENDERALGO_THROW_EXCEPTION("Invalid buffer type!");
        }
        }

        CHECK_GL_ERROR;

        return buffer;

    } catch (Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << "get inner buffer failed with exceptionL: " << e.what();
        assert(false);
        throw e;
    }
}

void RayCasterInnerResource::set_mask_label_level(LabelLevel level) {
    if (_label_level != level) {
        _label_level = level;
        if (GL_BASE == _gpu_platform) {
            memset(_inner_gl_resource->dirty_flag, 1, sizeof(bool) * TYPE_END);
            _shared_buffer_array.reset(new char[(int)_label_level * sizeof(Material)]);
        } else {
            memset(_inner_cuda_resource->dirty_flag, 1, sizeof(bool) * TYPE_END);
            _shared_buffer_array.reset(new char[(int)_label_level * 56]);
        }
    }
}

void RayCasterInnerResource::set_window_level(float ww, float wl, unsigned char label) {
    const Vector2f wl_v2(ww, wl);
    auto it = _window_levels.find(label);

    if (it == _window_levels.end()) {
        _window_levels.insert(std::make_pair(label, wl_v2));
        set_dirty(WINDOW_LEVEL_BUCKET);
    } else {
        if (wl_v2 != it->second) {
            it->second = wl_v2;
            set_dirty(WINDOW_LEVEL_BUCKET);
        }
    }
}

void RayCasterInnerResource::set_visible_labels(
    std::vector<unsigned char> labels) {
    if (_labels != labels) {
        _labels = labels;
        set_dirty(VISIBLE_LABEL_BUCKET);
        set_dirty(VISIBLE_LABEL_ARRAY);
    }
}

const std::vector<unsigned char>& RayCasterInnerResource::get_visible_labels() const {
    return _labels;
}

void RayCasterInnerResource::set_mask_overlay_color( std::map<unsigned char, RGBAUnit> colors) {
    if (_mask_overlay_colors != colors) {
        _mask_overlay_colors = colors;
        set_dirty(MASK_OVERLAY_COLOR_BUCKET);
    }
}

void RayCasterInnerResource::set_mask_overlay_color(const RGBAUnit& color, unsigned char label) {
    auto it = _mask_overlay_colors.find(label);

    if (it != _mask_overlay_colors.end()) {
        if (it->second != color) {
            it->second = color;
            set_dirty(MASK_OVERLAY_COLOR_BUCKET);
        }
    } else {
        _mask_overlay_colors[label] = color;
        set_dirty(MASK_OVERLAY_COLOR_BUCKET);
    }
}

const std::map<unsigned char, RGBAUnit>& RayCasterInnerResource::get_mask_overlay_color() const {
    return _mask_overlay_colors;
}

void RayCasterInnerResource::set_material(const Material& matrial, unsigned char label) {
    auto it = _material.find(label);

    if (it == _material.end()) {
        _material.insert(std::make_pair(label, matrial));
        set_dirty(MATERIAL_BUCKET);
    } else {
        if (matrial != it->second) {
            it->second = matrial;
            set_dirty(MATERIAL_BUCKET);
        }
    }
}

void RayCasterInnerResource::set_dirty(BufferType type) {
    if (GL_BASE == _gpu_platform) {
        _inner_gl_resource->dirty_flag[type] = true;
    }
    else {
        _inner_cuda_resource->dirty_flag[type] = true;
    }
}

void RayCasterInnerResource::remove_dirty(BufferType type) {
    if (GL_BASE == _gpu_platform) {
        _inner_gl_resource->dirty_flag[type] = false;
    }
    else {
        _inner_cuda_resource->dirty_flag[type] = false;
    }
}

bool RayCasterInnerResource::check_dirty(BufferType type) {
    if (GL_BASE == _gpu_platform) {
        return _inner_gl_resource->dirty_flag[type];
    }
    else {
        return _inner_cuda_resource->dirty_flag[type];
    }
}

namespace {
    inline int* get_visible_label_array(void* s_array, int label_level) {
        return (int*)(s_array);
    }
    inline size_t get_visible_label_offset(int label_level) {
        return 0;
    }

    inline float* get_wl_array(void* s_array, int label_level) {
        return (float*)((char*)(s_array) + 4 * label_level);
    }

    inline size_t get_wl_offset(int label_level) {
        return 4 * label_level;
    }

    inline cudaTextureObject_t* get_color_opacity_texture_array(void* s_array, int label_level) {
        return (cudaTextureObject_t*)((char*)(s_array) + 12 * label_level);
    }

    inline size_t get_color_opacity_texture_offset(int label_level) {
        return 12 * label_level;
    }

    inline float* get_material_array(void* s_array, int label_level) {
        return (float*)((char*)(s_array) + 20 * label_level);
    }

    inline size_t get_material_offset(int label_level) {
        return 20 * label_level;
    }
}

void RayCasterInnerResource::set_color_opacity_texture_array(GPUTexture1DArrayPairPtr tex_array) {
    _color_opacity_tex_array = tex_array;
}

CudaGlobalMemoryPtr RayCasterInnerResource::get_shared_map_memory() {
    if (CUDA_BASE != _gpu_platform) {
        RENDERALGO_THROW_EXCEPTION("invalid gpu platform.");
    }
    if (nullptr == _inner_cuda_resource->shared_map_memory) {
        _inner_cuda_resource->shared_map_memory = 
            CudaResourceManager::instance()->create_global_memory("ray-caster inner cuda resource");
    }

    CudaGlobalMemoryPtr shared_map_memory = _inner_cuda_resource->shared_map_memory;
    const int label_level = (int)_label_level;
    
    if (check_dirty(VISIBLE_LABEL_BUCKET)) {
        int* visible_labels = get_visible_label_array(_shared_buffer_array.get(), label_level);
        memset(visible_labels, 0, sizeof(int)*label_level);
        for (auto it = _labels.begin(); it != _labels.end(); ++it) {
            if (*it >= label_level) {
                RENDERALGO_THROW_EXCEPTION("visible label overflow.");
            }
            visible_labels[*it] = 1;
        }

        if (0 != shared_map_memory->update(get_visible_label_offset(label_level), sizeof(int)*label_level, visible_labels)) {
            RENDERALGO_THROW_EXCEPTION("update visible shared map memory failed.");
        }
        remove_dirty(VISIBLE_LABEL_BUCKET);                
    }

    if (check_dirty(WINDOW_LEVEL_BUCKET)) {
        float* wls = get_wl_array(_shared_buffer_array.get(), label_level);
        memset(wls, 0, sizeof(float)*2*label_level);
        unsigned char label = 0;
        for (auto it = _window_levels.begin(); it != _window_levels.end(); ++it) {
            label = it->first;
            if (label >= label_level) {
                RENDERALGO_THROW_EXCEPTION("window level label overflow.");
            }
            wls[label * 2] = it->second.get_x();
            wls[label * 2 + 1] = it->second.get_y();
        }

        if (0 != shared_map_memory->update(get_wl_offset(label_level), sizeof(float)*2*label_level, wls)) {
            RENDERALGO_THROW_EXCEPTION("update visible shared map memory failed.");
        }
        remove_dirty(WINDOW_LEVEL_BUCKET);
    }

    {
        //always update color opacity texture array(can't catch dirty)
        if (!_color_opacity_tex_array->cuda()) {
            RENDERALGO_THROW_EXCEPTION("invalid color opacity texture array platform.");
        }

        cudaTextureObject_t* color_opacity_tex_array = get_color_opacity_texture_array(_shared_buffer_array.get(), label_level);
        _color_opacity_tex_array->get_cuda_resource()->get_texture_array(
            cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeNormalizedFloat, true, color_opacity_tex_array);

        if (0 != shared_map_memory->update(get_color_opacity_texture_offset(label_level), sizeof(cudaTextureObject_t)*label_level, color_opacity_tex_array)) {
            RENDERALGO_THROW_EXCEPTION("update visible shared map memory failed.");
        }
    }

    if (check_dirty(MATERIAL_BUCKET)) {
        float* material_array = get_material_array(_shared_buffer_array.get(), label_level);
        memset(material_array, 0, sizeof(float)*9*label_level);
        unsigned char label = 0;
        for (auto it = _material.begin(); it != _material.end(); ++it) {
            label = it->first;
            if (label >= label_level) {
                RENDERALGO_THROW_EXCEPTION("material label overflow.");
            }
            Material& ma = it->second;
            material_array[label * 9]     = ma.diffuse[0];
            material_array[label * 9 + 1] = ma.diffuse[1];
            material_array[label * 9 + 2] = ma.diffuse[2];
            material_array[label * 9 + 3] = ma.diffuse[3];
            material_array[label * 9 + 4] = ma.specular[0];
            material_array[label * 9 + 5] = ma.specular[1];
            material_array[label * 9 + 6] = ma.specular[2];
            material_array[label * 9 + 7] = ma.specular[3];
            material_array[label * 9 + 8] = ma.specular_shiness;
        }

        if (0 != shared_map_memory->update(get_material_offset(label_level), sizeof(float)*9*label_level, material_array)) {
            RENDERALGO_THROW_EXCEPTION("update visible shared map memory failed.");
        }
    }

    return shared_map_memory;
}

MED_IMG_END_NAMESPACE
