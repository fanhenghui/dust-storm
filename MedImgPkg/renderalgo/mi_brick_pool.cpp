#include "mi_brick_pool.h"

#include <fstream>

#include "glresource/mi_gl_buffer.h"
#include "cudaresource/mi_cuda_resource_manager.h"
#include "cudaresource/mi_cuda_device_memory.h"

#include "io/mi_configure.h"
#include "io/mi_image_data.h"

#include "mi_brick_info_calculator.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

BrickPool::BrickPool(GPUPlatform p, unsigned int brick_size , unsigned int brick_margin):
    _gpu_platform(p),
    _brick_size(brick_size),
    _brick_margin(brick_margin),
    _brick_count(0),
    _volume_brick_info_cal(new VolumeBrickInfoCalculator(p)),
    _mask_brick_info_cal(new MaskBrickInfoCalculator(p)) {
    _brick_dim[0] = 1;
    _brick_dim[1] = 1;
    _brick_dim[2] = 1;
}

BrickPool::~BrickPool() {

}

void BrickPool::set_volume(std::shared_ptr<ImageData> image_data) {
    _volume = image_data;
}

void BrickPool::set_mask(std::shared_ptr<ImageData> image_data) {
    _mask = image_data;
}

void BrickPool::set_volume_texture(GPUTexture3DPairPtr tex) {
    _volume_texture = tex;
}

void BrickPool::set_mask_texture(GPUTexture3DPairPtr tex) {
    _mask_texture = tex;
}

void BrickPool::get_brick_dim(unsigned int (&brick_dim)[3]) {
    memcpy(brick_dim , _brick_dim , sizeof(unsigned int) * 3);
}

unsigned int BrickPool::get_brick_count() const {
    return _brick_count;
}

unsigned int BrickPool::get_brick_size() const {
    return _brick_size;
}

unsigned int BrickPool::get_brick_margin() const {
    return _brick_margin;
}

void BrickPool::calculate_volume_brick_info() {
    if (nullptr == _volume_brick_info_array) {
        _volume_brick_info_array.reset(new VolumeBrickInfo[this->get_brick_count()]);
    }

    if (GL_BASE == _gpu_platform) {
        if (nullptr == _volume_brick_info_buffer) {
            UIDType uid;
            GLBufferPtr buf = GLResourceManagerContainer::instance()->get_buffer_manager()->create_object(uid);
            buf->set_description("volume brick info buffer");
            buf->initialize();
            buf->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
            buf->bind();
            buf->load(this->get_brick_count() * sizeof(VolumeBrickInfo), NULL, GL_STATIC_DRAW);
            buf->unbind();
            _res_shield.add_shield<GLBuffer>(buf);

            _volume_brick_info_buffer.reset(new GPUMemoryPair(buf));
        }
    } else {
        //TODO CUDA
        if (nullptr == _volume_brick_info_buffer) {

        }
    }
    
    _volume_brick_info_cal->set_data(_volume);
    _volume_brick_info_cal->set_data_texture(_volume_texture);
    _volume_brick_info_cal->set_brick_info_array((char*)_volume_brick_info_array.get());
    _volume_brick_info_cal->set_brick_info_buffer(_volume_brick_info_buffer);
    _volume_brick_info_cal->set_brick_size(_brick_size);
    _volume_brick_info_cal->set_brick_margin(_brick_margin);
    _volume_brick_info_cal->set_brick_dim(_brick_dim);
    _volume_brick_info_cal->calculate();

    //Delete GPU memory
    if (GL_BASE == _gpu_platform) {
        _res_shield.remove_shield(_volume_brick_info_buffer->get_gl_resource());
    }
    _volume_brick_info_buffer = nullptr;
}

VolumeBrickInfo* BrickPool::get_volume_brick_info() const {
    return _volume_brick_info_array.get();
}

void BrickPool::write_volume_brick_info(const std::string& path) {
    RENDERALGO_CHECK_NULL_EXCEPTION(_volume_brick_info_array);
    std::ofstream out(path , std::ios::out);

    if (out.is_open()) {
        out << "volume dim : " << _volume->_dim[0] << " " << _volume->_dim[1] << " " << _volume->_dim[2] <<
            std::endl;
        out << "brick size : " << _brick_size << std::endl;
        out << "brick margin : " << _brick_margin << std::endl;
        out << "brick dim : " << _brick_dim[0] << " " << _brick_dim[1] << " " << _brick_dim[2] << std::endl;
        out << "brick count : " << _brick_count << std::endl;
        out << "brick info : id\tmin\tmax\n";

        for (unsigned int i = 0 ; i < _brick_count ; ++i) {
            out << i << "\t" << _volume_brick_info_array[i].min << "\t" << _volume_brick_info_array[i].max <<
                std::endl;
        }

        out.close();
    }
}

void BrickPool::add_visible_labels_cache(const std::vector<unsigned char>& vis_labels) {
    _vis_labels_cache[LabelKey(vis_labels)] = (vis_labels);
}

void BrickPool::get_visible_labels_cache(std::vector<std::vector<unsigned char>>& vis_labels) {
    vis_labels.clear();
    for (auto it = _vis_labels_cache.begin(); it != _vis_labels_cache.end(); ++it) {
        vis_labels.push_back(it->second);
    }
}

void BrickPool::clear_visible_labels_cache() {
    _vis_labels_cache.clear();
}

std::vector<std::vector<unsigned char>> BrickPool::get_stored_visible_labels() {
    std::vector<std::vector<unsigned char>> stored_labels;
    for (auto it = _mask_brick_info_array_set.begin(); it != _mask_brick_info_array_set.end(); ++it) {
        stored_labels.push_back(LabelKey::extract_labels(it->first));
    }
    return stored_labels;
}

void BrickPool::calculate_mask_brick_info(const std::vector<unsigned char>& vis_labels) {
    MI_RENDERALGO_LOG(MI_TRACE) << "IN brick pool calculate mask brick info";

    LabelKey key(vis_labels);
    MaskBrickInfo* brick_info = this->get_mask_brick_info(vis_labels);

    if (nullptr == brick_info) {
        std::unique_ptr<MaskBrickInfo[]> info_array(new MaskBrickInfo[this->get_brick_count()]);
        brick_info = info_array.get();
        _mask_brick_info_array_set.insert(std::make_pair(key, std::move(info_array)));

        if (GL_BASE == _gpu_platform) {
            UIDType uid;
            GLBufferPtr info_buffer = GLResourceManagerContainer::instance()->get_buffer_manager()->create_object(uid);
            info_buffer->set_description("mask brick info buffer " + key.key);
            info_buffer->initialize();
            info_buffer->set_buffer_target(GL_SHADER_STORAGE_BUFFER);
            info_buffer->bind();
            info_buffer->load(this->get_brick_count() * sizeof(MaskBrickInfo), NULL, GL_STATIC_DRAW);
            info_buffer->unbind();
            _res_shield.add_shield<GLBuffer>(info_buffer);
            _mask_brick_info_buffer_set.insert(std::make_pair(key, GPUMemoryPairPtr(new GPUMemoryPair(info_buffer))));
        } else {
            //TODO CUDA
        }
    }

    _mask_brick_info_cal->set_data(_mask);
    _mask_brick_info_cal->set_data_texture(_mask_texture);
    _mask_brick_info_cal->set_brick_info_array((char*)brick_info);
    _mask_brick_info_cal->set_brick_info_buffer(_mask_brick_info_buffer_set[key]);
    _mask_brick_info_cal->set_brick_size(_brick_size);
    _mask_brick_info_cal->set_brick_margin(_brick_margin);
    _mask_brick_info_cal->set_brick_dim(_brick_dim);
    _mask_brick_info_cal->set_visible_labels(vis_labels);
    _mask_brick_info_cal->calculate();

    MI_RENDERALGO_LOG(MI_TRACE) << "OUT brick pool calculate mask brick info";
}

void BrickPool::update_mask_brick_info(const AABBUI& aabb) {
    for (auto it = _mask_brick_info_array_set.begin() ; it != _mask_brick_info_array_set.end() ; ++it) {
        std::vector<unsigned char> vis_labels = LabelKey::extract_labels(it->first);
        _mask_brick_info_cal->set_data(_mask);
        _mask_brick_info_cal->set_data_texture(_mask_texture);
        _mask_brick_info_cal->set_brick_info_array((char*)it->second.get());
        _mask_brick_info_cal->set_brick_info_buffer(_mask_brick_info_buffer_set[it->first]);
        _mask_brick_info_cal->set_brick_size(_brick_size);
        _mask_brick_info_cal->set_brick_margin(_brick_margin);
        _mask_brick_info_cal->set_brick_dim(_brick_dim);
        _mask_brick_info_cal->set_visible_labels(vis_labels);
        _mask_brick_info_cal->update(aabb);
    }
}

MaskBrickInfo* BrickPool::get_mask_brick_info(const std::vector<unsigned char>& vis_labels) const {
    LabelKey key(vis_labels);
    auto it = _mask_brick_info_array_set.find(key);

    if (it == _mask_brick_info_array_set.end()) {
        return nullptr;
    } else {
        return it->second.get();
    }
}

void BrickPool::write_mask_brick_info(const std::string& path ,
                                      const std::vector<unsigned char>& visible_labels) {
    MaskBrickInfo* mask_brick_info = get_mask_brick_info(visible_labels);

    if (mask_brick_info) {
        std::ofstream out(path , std::ios::out);

        if (out.is_open()) {
            out << "volume dim : " << _mask->_dim[0] << " " << _mask->_dim[1] << " " << _mask->_dim[2] <<
                std::endl;
            out << "brick size : " << _brick_size << std::endl;
            out << "brick margin : " << _brick_margin << std::endl;
            out << "brick dim : " << _brick_dim[0] << " " << _brick_dim[1] << " " << _brick_dim[2] << std::endl;
            out << "brick count : " << _brick_count << std::endl;
            out << "brick info : id\tlabel\n";

            for (unsigned int i = 0 ; i < _brick_count ; ++i) {
                out << i << "\t" << mask_brick_info[i].label << std::endl;
            }

            out.close();
        }
    }
}

void BrickPool::remove_mask_brick_info(const std::vector<unsigned char>& vis_labels) {
    LabelKey key(vis_labels);
    auto it_array = _mask_brick_info_array_set.find(key);

    if (it_array != _mask_brick_info_array_set.end()) {
        _mask_brick_info_array_set.erase(it_array);
    }

    auto it_buffer = _mask_brick_info_buffer_set.find(key);

    if (it_buffer != _mask_brick_info_buffer_set.end()) {
        _mask_brick_info_buffer_set.erase(it_buffer);

        //release resource immediately
        if (GL_BASE == _gpu_platform) {
            _res_shield.remove_shield<GLBuffer>(it_buffer->second->get_gl_resource());
            GLResourceManagerContainer::instance()->get_buffer_manager()->
                remove_object(it_buffer->second->get_gl_resource());
        }
    }
}

void BrickPool::remove_all_mask_brick_info() {
    _mask_brick_info_array_set.clear();

    for (auto it_buffer = _mask_brick_info_buffer_set.begin() ;
            it_buffer != _mask_brick_info_buffer_set.end() ; ++it_buffer) {
        //release resource immediately
        if (GL_BASE == _gpu_platform) {
            _res_shield.remove_shield<GLBuffer>(it_buffer->second->get_gl_resource());
            GLResourceManagerContainer::instance()->get_buffer_manager()->
                remove_object(it_buffer->second->get_gl_resource());
        }
    }

    _mask_brick_info_buffer_set.clear();
}

void BrickPool::calculate_brick_geometry() {
    RENDERALGO_CHECK_NULL_EXCEPTION(_volume);

    for (int i = 0 ; i < 3 ; ++i) {
        _brick_dim[i] = (unsigned int)ceil((float)_volume->_dim[i] / (float)_brick_size);
    }

    _brick_count = _brick_dim[0] * _brick_dim[1] * _brick_dim[2];

    const unsigned int* dim = _volume->_dim;

    const unsigned int vertex_count = (_brick_dim[0] + 1) * (_brick_dim[1] + 1) * (_brick_dim[2] + 1);
    _brick_geometry.vertex_count = vertex_count;//vertex count is not the same with brick count

    _brick_geometry.vertex_array = new float[vertex_count * 3];
    _brick_geometry.color_array = new float[vertex_count * 4];
    _brick_geometry.brick_idx_units = new BrickEleIndex[_brick_count];


    float* vertex_array = _brick_geometry.vertex_array;
    float* color_array = _brick_geometry.color_array;
    BrickEleIndex* element_array = _brick_geometry.brick_idx_units;

    //vertex
    const float brick_size = static_cast<float>(_brick_size);
    float fx(0.0f), fy(0.0f), fz(0.0f);

    for (unsigned int z = 0 ; z < _brick_dim[2] + 1 ; ++z) {
        fz = (z == _brick_dim[2]) ? dim[2] : static_cast<float>(z) * brick_size;

        for (unsigned int y = 0 ; y < _brick_dim[1] + 1 ; ++y) {
            fy = (y == _brick_dim[1]) ?  dim[1] : static_cast<float>(y) * brick_size;

            for (unsigned int x = 0 ; x < _brick_dim[0] + 1 ; ++x) {
                fx = (x == _brick_dim[0]) ? dim[0] : static_cast<float>(x) * brick_size;
                const unsigned  int vertex_id = z * (_brick_dim[0] + 1) * (_brick_dim[1] + 1) + y *
                                                (_brick_dim[0] + 1) + x;
                vertex_array[vertex_id * 3] = fx;
                vertex_array[vertex_id * 3 + 1] = fy;
                vertex_array[vertex_id * 3 + 2] = fz;
            }
        }
    }

    //set vertex coordinate as color (not normalized)
    for (unsigned int i = 0; i < vertex_count ; ++i) {
        color_array[i * 4] = vertex_array[i * 3];
        color_array[i * 4 + 1] = vertex_array[i * 3 + 1];
        color_array[i * 4 + 2] = vertex_array[i * 3 + 2];
        color_array[i * 4 + 3] = 1.0;
    }

    //element
#define VertexID(pt0,pt1,pt2) (pt2[2]*(_brick_dim[0]+1)*(_brick_dim[1]+1) + pt1[1]*(_brick_dim[0]+1) + pt0[0] )

    for (unsigned int z = 0 ; z < _brick_dim[2] ; ++z) {
        for (unsigned int y = 0 ; y < _brick_dim[1] ; ++y) {
            for (unsigned int x = 0 ; x < _brick_dim[0] ; ++x) {
                const unsigned int idx = z * _brick_dim[0] * _brick_dim[1] + y * _brick_dim[0] + x;
                const unsigned int ptmin[3] = {x , y , z};
                const unsigned int ptmax[3] = {x + 1 , y + 1 , z + 1};

                unsigned int* pIdx = element_array[idx].idx;
                pIdx[0] = VertexID(ptmax, ptmin, ptmax);
                pIdx[1] = VertexID(ptmax, ptmin, ptmin);
                pIdx[2] = VertexID(ptmax, ptmax, ptmin);
                pIdx[3] = VertexID(ptmax, ptmin, ptmax);
                pIdx[4] = VertexID(ptmax, ptmax, ptmin);
                pIdx[5] = VertexID(ptmax, ptmax, ptmax);
                pIdx[6] = VertexID(ptmax, ptmax, ptmax);
                pIdx[7] = VertexID(ptmax, ptmax, ptmin);
                pIdx[8] = VertexID(ptmin, ptmax, ptmin);
                pIdx[9] = VertexID(ptmax, ptmax, ptmax);
                pIdx[10] = VertexID(ptmin, ptmax, ptmin);
                pIdx[11] = VertexID(ptmin, ptmax, ptmax);
                pIdx[12] = VertexID(ptmax, ptmax, ptmax);
                pIdx[13] = VertexID(ptmin, ptmax, ptmax);
                pIdx[14] = VertexID(ptmin, ptmin, ptmax);
                pIdx[15] = VertexID(ptmax, ptmax, ptmax);
                pIdx[16] = VertexID(ptmin, ptmin, ptmax);
                pIdx[17] = VertexID(ptmax, ptmin, ptmax);
                pIdx[18] = VertexID(ptmin, ptmin, ptmax);
                pIdx[19] = VertexID(ptmin, ptmax, ptmax);
                pIdx[20] = VertexID(ptmin, ptmax, ptmin);
                pIdx[21] = VertexID(ptmin, ptmin, ptmax);
                pIdx[22] = VertexID(ptmin, ptmax, ptmin);
                pIdx[23] = VertexID(ptmin, ptmin, ptmin);
                pIdx[24] = VertexID(ptmin, ptmin, ptmax);
                pIdx[25] = VertexID(ptmin, ptmin, ptmin);
                pIdx[26] = VertexID(ptmax, ptmin, ptmin);
                pIdx[27] = VertexID(ptmin, ptmin, ptmax);
                pIdx[28] = VertexID(ptmax, ptmin, ptmin);
                pIdx[29] = VertexID(ptmax, ptmin, ptmax);
                pIdx[30] = VertexID(ptmin, ptmin, ptmin);
                pIdx[31] = VertexID(ptmin, ptmax, ptmin);
                pIdx[32] = VertexID(ptmax, ptmax, ptmin);
                pIdx[33] = VertexID(ptmin, ptmin, ptmin);
                pIdx[34] = VertexID(ptmax, ptmax, ptmin);
                pIdx[35] = VertexID(ptmax, ptmin, ptmin);
            }
        }
    }

#undef VertexID
}

const BrickGeometry& BrickPool::get_brick_geometry() const {
    return _brick_geometry;
}

void BrickPool::calculate_intercect_brick_range(const AABB& bounding , AABBI& brick_range) {
    const float brick_size = float(_brick_size);
    float range_min[3] = {
        (float)bounding._min.x / brick_size,
        (float)bounding._min.y / brick_size,
        (float)bounding._min.z / brick_size
    };

    float range_max[3] = {
        (float)bounding._max.x / brick_size,
        (float)bounding._max.y / brick_size,
        (float)bounding._max.z / brick_size
    };

    for (int i = 0 ; i < 3 ; ++i) {
        if (fabs(range_min[i] - int(range_min[i])) < 1e-6f) {
            range_min[i] = float(int(range_min[i]));
            brick_range._min[i] = (int)floor(range_min[i]);
        } else {
            brick_range._min[i] = (int)floor(range_min[i]);
        }

        if (fabs(range_max[i] - int(range_max[i])) < 1e-6f) {
            range_max[i] = float(int(range_max[i]));
            brick_range._max[i] = (int)floor(range_max[i]) - 1;
        } else {
            brick_range._max[i] = (int)floor(range_max[i]);
        }
    }
}

void BrickPool::get_clipping_brick_geometry(const AABB& bounding, float* brick_vertex,
        float* brick_color) {
    RENDERALGO_CHECK_NULL_EXCEPTION(brick_vertex);
    RENDERALGO_CHECK_NULL_EXCEPTION(brick_color);

    //get intersect brick index range
    AABBI brick_range;
    calculate_intercect_brick_range(bounding , brick_range);

    memcpy(brick_vertex , _brick_geometry.vertex_array,
           sizeof(float) * 3 * _brick_geometry.vertex_count);
    memcpy(brick_color , _brick_geometry.color_array ,
           sizeof(float) * 4 * _brick_geometry.vertex_count);

    //Change 6 plane vertex coordinate
    //Vertex ID minus brick ID is 1. So here vertex ID should plus 1 , otherwise can cause a slope in the edge
    const int brick_dim_layout_jump = (_brick_dim[0] + 1) * (_brick_dim[1] + 1);
    int idx_min(0), idx_max(0);
    int change_min(0), change_max(0);
    int iYJump(0), z_jump(0);

    float bounding_min[3] = {
        (float)bounding._min.x ,
        (float)bounding._min.y ,
        (float)bounding._min.z
    };

    float bounding_max[3] = {
        (float)bounding._max.x,
        (float)bounding._max.y,
        (float)bounding._max.z
    };

    //Plane X
    change_min = brick_range._min[0];
    change_max = brick_range._max[0] + 1;

    for (int z = brick_range._min[2] ; z <= brick_range._max[2] + 1 ; ++z) {
        z_jump = z * brick_dim_layout_jump;

        for (int y = brick_range._min[1] ; y <= brick_range._max[1] + 1 ; ++y) {
            idx_min = z_jump + y * (_brick_dim[0] + 1) + change_min;
            idx_max = z_jump + y * (_brick_dim[0] + 1) + change_max;

            brick_vertex[idx_min * 3 + 0] = bounding_min[0];
            brick_vertex[idx_max * 3 + 0] = bounding_max[0];
            brick_color[idx_min * 4 + 0] = bounding_min[0];
            brick_color[idx_max * 4 + 0] = bounding_max[0];
        }
    }

    //Plane Y
    change_min = brick_range._min[1];
    change_max = brick_range._max[1] + 1;

    for (int z = brick_range._min[2] ; z <= brick_range._max[2] + 1 ; ++z) {
        z_jump = z * brick_dim_layout_jump;

        for (int x = brick_range._min[0] ; x <= brick_range._max[0] + 1 ; ++x) {
            idx_min = z_jump + change_min * (_brick_dim[0] + 1) + x;
            idx_max = z_jump + change_max * (_brick_dim[0] + 1) + x;

            brick_vertex[idx_min * 3 + 1] = bounding_min[1];
            brick_vertex[idx_max * 3 + 1] = bounding_max[1];
            brick_color[idx_min * 4 + 1] = bounding_min[1];
            brick_color[idx_max * 4 + 1] = bounding_max[1];
        }
    }

    //Plane Z
    change_min = brick_range._min[2];
    change_max = brick_range._max[2] + 1;

    for (int y = brick_range._min[1] ; y <= brick_range._max[1] + 1 ; ++y) {
        iYJump = y * (_brick_dim[0] + 1);

        for (int x = brick_range._min[0] ; x <= brick_range._max[0] + 1 ; ++x) {
            idx_min = change_min * brick_dim_layout_jump + iYJump + x;
            idx_max = change_max * brick_dim_layout_jump + iYJump + x;

            brick_vertex[idx_min * 3 + 2] = bounding_min[2];
            brick_vertex[idx_max * 3 + 2] = bounding_max[2];
            brick_color[idx_min * 4 + 2] = bounding_min[2];
            brick_color[idx_max * 4 + 2] = bounding_max[2];
        }
    }
}

void BrickPool::debug_save_mask_info(const std::string& path)
{
    for (auto it = _mask_brick_info_array_set.begin(); it != _mask_brick_info_array_set.end(); 
        ++it) {
        std::vector<unsigned char> labels = LabelKey::extract_labels(it->first);
        std::stringstream ss;
        ss << "mask";
        for (auto l = labels.begin(); l != labels.end(); ++l) {
            ss << "_" << static_cast<unsigned int>(*l);
        }
        ss << ".txt";
        const std::string file_name = path + "/" + ss.str();
        std::ofstream out(file_name.c_str(), std::ios::out);
        if(!out.is_open()) {
            MI_RENDERALGO_LOG(MI_WARNING) << "open file " << file_name << " to debug save mask info failed.";
            continue;
        }

        MaskBrickInfo* info = it->second.get();
        out << "volume dim "  << _volume->_dim[0] << " " << _volume->_dim[1] << " " << _volume->_dim[2] << std::endl;
        out << "brick dim " << _brick_dim[0] << " " << _brick_dim[1] << " " << " " << _brick_dim[2] << std::endl;
        for (unsigned int i = 0; i < _brick_count; ++i ) {
            out << i << " _" << info[i].label << std::endl;
        }
        out.close();
        MI_RENDERALGO_LOG(MI_WARNING) << "save file " << file_name << " to debug save mask info done.";
    }
}

MED_IMG_END_NAMESPACE