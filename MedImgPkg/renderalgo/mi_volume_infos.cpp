#include "mi_volume_infos.h"

#include "io/mi_configure.h"

#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "glresource/mi_gl_resource_manager_container.h"
#include "glresource/mi_gl_texture_3d.h"
#include "glresource/mi_gl_utils.h"

#include "mi_brick_info_calculator.h"
#include "mi_brick_pool.h"
#include "mi_camera_calculator.h"

#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

namespace {

template <typename SrcType, typename DstType>
std::unique_ptr<DstType[]> signed_to_unsigned(unsigned int length,
        double min_gray, void* data_src) {
    std::unique_ptr<DstType[]> data_dst(new DstType[length]);
    SrcType* data_src0 = (SrcType*)(data_src);

    for (unsigned int i = 0; i < length; ++i) {
        data_dst[i] =
            static_cast<DstType>(static_cast<double>(data_src0[i]) - min_gray);
    }

    return std::move(data_dst);
}
}

VolumeInfos::VolumeInfos(GPUPlatform p) : _gpu_platform(p), _volume_dirty(false), _mask_dirty(false) {
    _brick_pool.reset(new BrickPool(p, 16, 2));
}

VolumeInfos::~VolumeInfos() {
    finialize();
}

void VolumeInfos::finialize() {
    if (GL_BASE == _gpu_platform) {
        if (nullptr != _volume_texture) {
            GLResourceManagerContainer::instance()->get_texture_3d_manager()->
                remove_object(_volume_texture->get_gl_resource());
        }

        if (nullptr != _mask_texture) {
            GLResourceManagerContainer::instance()->get_texture_3d_manager()->
                remove_object(_mask_texture->get_gl_resource());       
        }
    }

    _volume_texture = nullptr;
    _mask_texture = nullptr;
}

void VolumeInfos::set_volume(std::shared_ptr<ImageData> image_data) {
    try {
        RENDERALGO_CHECK_NULL_EXCEPTION(image_data);

        _volume_data = image_data;
        _volume_dirty = true;

        if (GL_BASE == _gpu_platform) {
            // release textures
            if (nullptr != _volume_texture) {
                GLResourceManagerContainer::instance()->get_texture_3d_manager()->
                    remove_object(_volume_texture->get_gl_resource());
            }

            // create texture
            UIDType uid(0);
            GLTexture3DPtr tex = GLResourceManagerContainer::instance()->get_texture_3d_manager()->create_object(uid);
            if (_data_header) {
                tex->set_description("volume : " + _data_header->series_uid);
            }
            else {
                tex->set_description("volume : undefined series UID");
            }
            _volume_texture.reset(new GPUTexture3DPair(tex));
        } else {
            //TODO CUDA
        }

        // brick pool
        _brick_pool->set_volume(_volume_data);
        _brick_pool->set_volume_texture(_volume_texture);

        // Create camera calculator
        _camera_calculator.reset(new CameraCalculator(_volume_data));

    } catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << "set volume failed with exception: " << e.what();
        assert(false);
    }
}

void VolumeInfos::set_mask(std::shared_ptr<ImageData> image_data) {
    try {
        RENDERALGO_CHECK_NULL_EXCEPTION(image_data);

        _mask_data = image_data;
        _mask_dirty = true;

        if (GL_BASE == _gpu_platform) {
            // release textures
            if (nullptr != _mask_texture) {
                GLResourceManagerContainer::instance()->get_texture_3d_manager()->
                    remove_object(_mask_texture->get_gl_resource());
            }

            // create texture
            UIDType uid(0);
            GLTexture3DPtr tex = GLResourceManagerContainer::instance()->get_texture_3d_manager()->create_object(uid);
            if (_data_header) {
                tex->set_description("mask : " + _data_header->series_uid);
            }
            else {
                tex->set_description("mask : undefined series UID");
            }
            _mask_texture.reset(new GPUTexture3DPair(tex));
        } else {
            //TODO CUDA
        }

        // brick pool
        _brick_pool->set_mask(_mask_data);
        _brick_pool->set_mask_texture(_mask_texture);
    } catch (const Exception& e) {
        MI_RENDERALGO_LOG(MI_FATAL) << "set mask failed with exception: " << e.what();
        assert(false);
    }
}

void VolumeInfos::set_data_header(
    std::shared_ptr<ImageDataHeader> data_header) {
    _data_header = data_header;
}

GPUTexture3DPairPtr VolumeInfos::get_volume_texture() {
    return _volume_texture;
}

GPUTexture3DPairPtr VolumeInfos::get_mask_texture() {
    return _mask_texture;
}

std::shared_ptr<ImageData> VolumeInfos::get_volume() {
    return _volume_data;
}

std::shared_ptr<ImageData> VolumeInfos::get_mask() {
    return _mask_data;
}

void VolumeInfos::update_mask(const unsigned int (&begin)[3],
                              const unsigned int (&end)[3],
                              unsigned char* data_updated,
                              bool has_data_array_changed /*= true*/) {
    // update mask CPU
    const unsigned int dim_brick[3] = { end[0] - begin[0], 
                                        end[1] - begin[1],
                                        end[2] - begin[2]};

    if (!has_data_array_changed) {
        unsigned char* mask_array = (unsigned char*)_mask_data->get_pixel_pointer();
        const unsigned int layer_whole = _mask_data->_dim[0] * _mask_data->_dim[1];
        const unsigned int layer_brick = dim_brick[0] * dim_brick[1];

        for (unsigned int z = begin[2]; z < end[2]; ++z) {
            for (unsigned int y = begin[1]; y < end[1]; ++y) {
                memcpy(mask_array + z * layer_whole + y * _mask_data->_dim[0] + begin[0],
                       data_updated + (z - begin[2]) * layer_brick +
                       (y - begin[1]) * dim_brick[0],
                       dim_brick[0]);
            }
        }
    }

    _mask_aabb_to_be_update.push_back(AABBUI(begin, end));
    _mask_array_to_be_update.push_back(data_updated);
}

void VolumeInfos::refresh_update_mask() {
    if (_mask_aabb_to_be_update.empty()) {
        return;
    }

    if (GL_BASE == _gpu_platform) {
        CHECK_GL_ERROR;

        unsigned int dim_brick[3]={0,0,0};
        GLUtils::set_pixel_pack_alignment(1);
        GLUtils::set_pixel_unpack_alignment(1);
        _mask_texture->get_gl_resource()->bind();

        for (size_t i = 0; i < _mask_aabb_to_be_update.size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                dim_brick[j] = _mask_aabb_to_be_update[i]._max[j] -
                               _mask_aabb_to_be_update[i]._min[j];
            }

            _mask_texture->get_gl_resource()->update(
                _mask_aabb_to_be_update[i]._min[0],
                _mask_aabb_to_be_update[i]._min[1],
                _mask_aabb_to_be_update[i]._min[2], dim_brick[0], dim_brick[1],
                dim_brick[2], GL_RED, GL_UNSIGNED_BYTE, _mask_array_to_be_update[i]);
            CHECK_GL_ERROR;

            delete[] _mask_array_to_be_update[i];
        }

        _mask_texture->get_gl_resource()->unbind();
        _mask_aabb_to_be_update.clear();
        _mask_array_to_be_update.clear();

        refresh_stored_mask_brick_info();
    } else {
        //TODO CUDA
    }
}

void VolumeInfos::refresh_cache_mask_brick_info() {
    std::vector<std::vector<unsigned char>> vis_labels;
    _brick_pool->get_visible_labels_cache(vis_labels);
    for (auto it = vis_labels.begin(); it != vis_labels.end(); ++it) {
        _brick_pool->calculate_mask_brick_info(*it);
    }
    _brick_pool->clear_visible_labels_cache();
}

void VolumeInfos::refresh_stored_mask_brick_info() {
    std::vector<std::vector<unsigned char>> stored_labels = _brick_pool->get_stored_visible_labels();
    for (auto it = stored_labels.begin(); it != stored_labels.end(); ++it) {
        _brick_pool->calculate_mask_brick_info(*it);
    }
}

void VolumeInfos::refresh_upload_mask() {
    if (!_mask_dirty) {
        return;
    }

    if (GL_BASE == _gpu_platform) {
        CHECK_GL_ERROR;
        GLTexture3DPtr& tex = _mask_texture->get_gl_resource();
        tex->initialize();
        tex->bind();
        GLTextureUtils::set_3d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_3D, GL_NEAREST);
        tex->load(GL_R8, _mask_data->_dim[0], _mask_data->_dim[1], _mask_data->_dim[2], 
            GL_RED, GL_UNSIGNED_BYTE, _mask_data->get_pixel_pointer());
        tex->unbind();
        CHECK_GL_ERROR;
    } else {
        //TODO CUDA
    }
    
    refresh_stored_mask_brick_info();
    _mask_dirty = false;
}

void VolumeInfos::refresh_upload_volume() {
    if (!_volume_dirty) {
        return;
    }

    if (GL_BASE == _gpu_platform) {
        CHECK_GL_ERROR;
        GLTexture3DPtr& tex = _volume_texture->get_gl_resource();
        tex->initialize();
        tex->bind();
        GLTextureUtils::set_3d_wrap_s_t_r(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_3D, GL_LINEAR);
        GLenum internal_format, format, type;
        GLUtils::get_gray_texture_format(_volume_data->_data_type, internal_format,
            format, type);

        const unsigned int length =
            _volume_data->_dim[0] * _volume_data->_dim[1] * _volume_data->_dim[2];
        const double min_gray = _volume_data->get_min_scalar();

        // signed integer should convert to unsigned
        if (_volume_data->_data_type == SHORT) {
            std::unique_ptr<unsigned short[]> dst_data =
                signed_to_unsigned<short, unsigned short>(
                    length, min_gray, _volume_data->get_pixel_pointer());
            tex->load(internal_format, _volume_data->_dim[0], _volume_data->_dim[1],
                _volume_data->_dim[2], format, type, dst_data.get());
        }
        else if (_volume_data->_data_type == CHAR) {
            std::unique_ptr<unsigned char[]> dst_data =
                signed_to_unsigned<char, unsigned char>(
                    length, min_gray, _volume_data->get_pixel_pointer());
            tex->load(internal_format, _volume_data->_dim[0], _volume_data->_dim[1],
                _volume_data->_dim[2], format, type, dst_data.get());
        }
        else {
            tex->load(internal_format, _volume_data->_dim[0], _volume_data->_dim[1],
                _volume_data->_dim[2], format, type,
                _volume_data->get_pixel_pointer());
        }

        tex->unbind();
        CHECK_GL_ERROR;
    } else {
        //TODO CUDA
    }

    _brick_pool->calculate_brick_geometry();
    _brick_pool->calculate_volume_brick_info();

    _volume_dirty = false;
}

std::shared_ptr<ImageDataHeader> VolumeInfos::get_data_header() {
    return _data_header;
}

std::shared_ptr<BrickPool> VolumeInfos::get_brick_pool() {
    return _brick_pool;
}

std::shared_ptr<CameraCalculator> VolumeInfos::get_camera_calculator() {
    return _camera_calculator;
}

void VolumeInfos::refresh() {
    if (Configure::instance()->get_processing_unit_type() == CPU) {
        return;
    }

    refresh_upload_volume();

    refresh_upload_mask();

    refresh_update_mask();

    refresh_cache_mask_brick_info();
}

void VolumeInfos::cache_original_mask() {
    if (nullptr == _mask_data) {
        MI_RENDERALGO_LOG(MI_ERROR) << "cache empty mask.";
        return;
    }
    _cache_original_mask.reset(new ImageData());
    _mask_data->deep_copy(_cache_original_mask.get());
}

std::shared_ptr<ImageData> VolumeInfos::get_cache_original_mask() {
    return _cache_original_mask;
}

MED_IMG_END_NAMESPACE