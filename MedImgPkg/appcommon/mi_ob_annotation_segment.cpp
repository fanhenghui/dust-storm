#include "mi_ob_annotation_segment.h"

#include "util/mi_configuration.h"

#include "arithmetic/mi_segment_threshold.h"
#include "arithmetic/mi_connected_domain_analysis.h"
#include "arithmetic/mi_intersection_test.h"

#include "io/mi_image_data.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_opacity_transfer_function.h"
#include "renderalgo/mi_transfer_function_loader.h"
#include "renderalgo/mi_mask_label_store.h"

#include "mi_model_annotation.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

OBAnnotationSegment::OBAnnotationSegment() {

}

OBAnnotationSegment::~OBAnnotationSegment() {

}

void OBAnnotationSegment::set_model(std::shared_ptr<ModelAnnotation> model) {
    _model = model;
}

void OBAnnotationSegment::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos) {
    _volume_infos = volume_infos;
}

void OBAnnotationSegment::set_mpr_scenes(std::vector<std::shared_ptr<MPRScene>> scenes) {
    _mpr_scenes = scenes;
}

void OBAnnotationSegment::set_vr_scenes(std::vector<std::shared_ptr<VRScene>> scenes) {
    _vr_scenes = scenes;
}

void OBAnnotationSegment::update(int code_id /*= 0*/) {
    APPCOMMON_CHECK_NULL_EXCEPTION(_volume_infos);

    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    APPCOMMON_CHECK_NULL_EXCEPTION(volume_data);

    std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();
    APPCOMMON_CHECK_NULL_EXCEPTION(camera_cal);

    std::shared_ptr<ModelAnnotation> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);

    //update overlay mask mode
    if (ModelAnnotation::ADD == code_id || ModelAnnotation::DELETE == code_id) {
        if (model->get_annotations().empty()) {
            //for (auto it = _mpr_scenes.begin(); it != _mpr_scenes.end(); ++it) {
            //    (*it)->set_mask_overlay_mode(MASK_OVERLAY_DISABLE);
            //}
         } else {
             for (auto it = _mpr_scenes.begin(); it != _mpr_scenes.end(); ++it) {
                 (*it)->set_mask_overlay_mode(MASK_OVERLAY_ENABLE);
            }
        }
    }

    /// \ ADD
    if (ModelAnnotation::ADD == code_id) {
        //get added annotations
        const std::map<std::string, ModelAnnotation::AnnotationUnit>& vois = model->get_annotations();
        typedef std::map<std::string, ModelAnnotation::AnnotationUnit>::const_iterator ConstVOIIter;
        std::vector<ConstVOIIter> vois_add;
        for (ConstVOIIter it = vois.begin() ; it != vois.end(); ++it) {
            const std::string id = it->first;
            if (_pre_vois.find(id) == _pre_vois.end()) {
                vois_add.push_back(it);
            }
        }

        //add pre_vois and segment
        std::vector<unsigned char> add_labels;
        for (auto it = vois_add.begin(); it != vois_add.end(); ++it) {
            const std::string id = (*it)->first;
            const VOISphere& voi = (*it)->second.voi;
            const unsigned char label = (*it)->second.label;
            add_labels.push_back(label);
            if (voi.diameter > 0.1f) {
                Ellipsoid ellipsoid = voi_patient_to_volume(voi);
                AABBUI aabb;
                get_aabb_i(ellipsoid, aabb);
                segment_i(ellipsoid , aabb , label);
                _pre_vois.insert(std::make_pair(id, VOIUnit(voi, label, aabb)));  
            } else {
                _pre_vois.insert(std::make_pair(id, VOIUnit(voi, label)));  
            }
        }

        //add visible parameter to scene
        for (auto it = _mpr_scenes.begin(); it != _mpr_scenes.end(); ++it) {
            std::vector<unsigned char> vis_labels = (*it)->get_visible_labels();
            for (auto it2 = add_labels.begin(); it2 != add_labels.end(); ++it2) {
                const unsigned char label = *it2;
                vis_labels.push_back(label);
                (*it)->set_mask_overlay_color(RGBAUnit(255.0f,0.0f,0.0f) , label);
            }
            (*it)->set_visible_labels(vis_labels);
        }

        if(!_vr_scenes.empty()) {
            const std::string color_opacity_xml = "../config/lut/3d/ct_lung_nodule.xml";
            std::shared_ptr<ColorTransFunc> color;
            std::shared_ptr<OpacityTransFunc> opacity;
            float ww, wl;
            RGBAUnit background;
            Material material;
            if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(color_opacity_xml, color, opacity, ww, wl, background, material)) {
                MI_APPCOMMON_LOG(MI_ERROR) << "load lut: " << color_opacity_xml << " failed.";
            } else {
                for (auto it = _vr_scenes.begin(); it != _vr_scenes.end(); ++it) {
                    std::vector<unsigned char> vis_labels = (*it)->get_visible_labels();
                    for (auto it2 = add_labels.begin(); it2 != add_labels.end(); ++it2) {
                        const unsigned char label = *it2;
                        vis_labels.push_back(label);
                        (*it)->set_color_opacity(color, opacity, label);
                        (*it)->set_material(material, label);
                        (*it)->set_window_level(ww, wl, label);
                    }
                    (*it)->set_visible_labels(vis_labels);
                }
            }
        }
    }

    /// \Delete
    if (ModelAnnotation::DELETE == code_id) {
        if (_pre_vois.empty()) {
            return;
        }
        //get deleted annotations
        const std::map<std::string, ModelAnnotation::AnnotationUnit>& vois = model->get_annotations();
        typedef std::map<std::string, OBAnnotationSegment::VOIUnit>::const_iterator ConstPreVOIIter;
        std::vector<ConstPreVOIIter> vois_delete;
        for (auto it = _pre_vois.begin() ; it != _pre_vois.end(); ++it) {
            const std::string id = it->first;
            if (vois.find(id) == vois.end()) {
                vois_delete.push_back(it);
            }
        }

        //update pre_vois and recover segmentation
        for (auto it = vois_delete.begin(); it != vois_delete.end(); ++it) {
            const std::string id = (*it)->first;
            const AABBUI& aabb = (*it)->second.aabb;
            const unsigned char label = (*it)->second.label;
            recover_i(aabb , label);
            _pre_vois.erase(*it);
        }

        for (auto it = _mpr_scenes.begin(); it != _mpr_scenes.end(); ++it) {
            (*it)->set_dirty(true);
        }
        for (auto it = _vr_scenes.begin(); it != _vr_scenes.end(); ++it) {
            (*it)->set_dirty(true);
        }
    }

    /// \Modifying
    if (ModelAnnotation::MODIFYING == code_id) {
        //do nothing
    }

    /// \Modify completed
    if (ModelAnnotation::MODIFY_COMPLETED == code_id) {
        const std::map<std::string, ModelAnnotation::AnnotationUnit>& vois = model->get_annotations();
        typedef std::map<std::string, ModelAnnotation::AnnotationUnit>::const_iterator ConstIter;
        typedef std::map<std::string, OBAnnotationSegment::VOIUnit>::iterator PreVOIIter;

        if(_pre_vois.size() != vois.size()) {
            MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation cache.";
            return;
        }

        //compare to find modified ones
        std::vector<ConstIter> vois_modify;
        std::vector<PreVOIIter> prevois_modify;
        for (ConstIter it = vois.begin(); it != vois.end() ; ++it) {
            const std::string id = it->first;
            const VOISphere& voi = it->second.voi;
            PreVOIIter it2 = _pre_vois.find(id);
            if (it2 == _pre_vois.end()) {
                MI_APPCOMMON_LOG(MI_ERROR) << "find modified voi " << id << " failed.";
                return;
            }
            const VOISphere& prevoi = it2->second.voi;
            if (voi != prevoi) {
                vois_modify.push_back(it);
                prevois_modify.push_back(it2);
            }
        }

        //modify segment
        for (size_t i = 0; i < vois_modify.size(); ++i) {

            const unsigned char label = vois_modify[i]->second.label;
            const VOISphere voi = vois_modify[i]->second.voi;
            //recover
            recover_i(prevois_modify[i]->second.aabb , label);
            //segment
            Ellipsoid ellipsoid = voi_patient_to_volume(voi);
            AABBUI aabb;
            get_aabb_i(ellipsoid, aabb);
            segment_i(ellipsoid , aabb , label);

            prevois_modify[i]->second.voi = voi;
            prevois_modify[i]->second.aabb = aabb;

            for (auto it = _mpr_scenes.begin(); it != _mpr_scenes.end(); ++it) {
                (*it)->set_dirty(true);
            }
            for (auto it = _vr_scenes.begin(); it != _vr_scenes.end(); ++it) {
                (*it)->set_dirty(true);
            }
        }
    }
}

Ellipsoid OBAnnotationSegment::voi_patient_to_volume(const VOISphere& voi) {
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();

    const Matrix4& mat_p2w = camera_cal->get_patient_to_world_matrix();
    const Matrix4& mat_w2v = camera_cal->get_world_to_volume_matrix();
    Matrix4 mat_p2v = mat_w2v*mat_p2w;

    PatientAxisInfo head_info = camera_cal->get_head_patient_axis_info();
    PatientAxisInfo posterior_info = camera_cal->get_posterior_patient_axis_info();
    PatientAxisInfo left_info = camera_cal->get_left_patient_axis_info();
    double basic_abc[3];
    basic_abc[head_info.volume_coord/2] = volume_data->_spacing[head_info.volume_coord/2];
    basic_abc[posterior_info.volume_coord/2] = volume_data->_spacing[posterior_info.volume_coord/2];
    basic_abc[left_info.volume_coord/2] = volume_data->_spacing[left_info.volume_coord/2];

    Ellipsoid ellipsoid;
    ellipsoid._center = mat_p2v.transform(voi.center);
    double voi_abc[3] = {0,0,0};
    voi_abc[head_info.volume_coord/2] = voi.diameter*0.5/basic_abc[head_info.volume_coord/2] ;
    voi_abc[left_info.volume_coord/2] = voi.diameter*0.5/basic_abc[left_info.volume_coord/2] ;
    voi_abc[posterior_info.volume_coord/2] = voi.diameter*0.5/basic_abc[posterior_info.volume_coord/2] ;
    ellipsoid._a = voi_abc[0];
    ellipsoid._b = voi_abc[1];
    ellipsoid._c = voi_abc[2];

    return ellipsoid;
}

int OBAnnotationSegment::get_aabb_i(const Ellipsoid& ellipsoid, AABBUI& aabb) {
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    unsigned int begin[3] , end[3];
    int inter_status = ArithmeticUtils::get_valid_region(volume_data->_dim , ellipsoid , begin , end);
    aabb = AABBUI(begin , end);
    return inter_status;
}

void OBAnnotationSegment::recover_i(const AABBUI& aabb , unsigned char label)
{
    std::shared_ptr<ImageData> mask_data = _volume_infos->get_mask();
    unsigned char* mask_array = (unsigned char*)mask_data->get_pixel_pointer();
    const unsigned int layer = mask_data->_dim[0]*mask_data->_dim[1];

#ifndef _DEBUG
#pragma omp parallel for
#endif
    for (unsigned int z = aabb._min[2] ; z < aabb._max[2] ; ++z) {
#ifndef _DEBUG
#pragma omp parallel for
#endif
        for (unsigned int y = aabb._min[1] ; y < aabb._max[1] ; ++y) {
#ifndef _DEBUG
#pragma omp parallel for
#endif
            for (unsigned int x = aabb._min[0] ; x < aabb._max[0] ; ++x) {
                unsigned int idx = z*layer + y*mask_data->_dim[0] + x;
                if (mask_array[idx] == label) {
                    mask_array[idx] = 0;
                }
            }
        }
    }

    //Update to texture
    if (GPU == Configuration::instance()->get_processing_unit_type()) {
        if (aabb != AABBUI()) {
            update_aabb_i(aabb);
        }
    }
}

void OBAnnotationSegment::segment_i(const Ellipsoid& ellipsoid , const AABBUI& aabb ,unsigned char label)
{
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    std::shared_ptr<ImageData> mask_data = _volume_infos->get_mask();
    const DataType data_type = volume_data->_data_type;

    unsigned int begin[3] = {0,0,0};
    unsigned int end[3] = {0,0,0};
    ArithmeticUtils::get_valid_region(volume_data->_dim, ellipsoid, begin, end);

    switch(data_type) {
    case SHORT: {
            //get threshold
            SegmentThreshold<short> segment;
            segment.set_data_ref((short*)volume_data->get_pixel_pointer());
            segment.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
            segment.set_dim(volume_data->_dim);
            segment.set_target_label(label);
            segment.set_min_scalar(volume_data->get_min_scalar());
            segment.set_max_scalar(volume_data->get_max_scalar());
            segment.segment_auto_threshold(ellipsoid , SegmentThreshold<short>::Otsu);

            ConnectedDomainAnalysis cd_analy;
            cd_analy.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
            cd_analy.set_dim(volume_data->_dim);
            cd_analy.set_target_label(label);
            cd_analy.set_roi(aabb._min , aabb._max);
            cd_analy.keep_major();
            break;
        }
    case USHORT: {
            //get threshold
            SegmentThreshold<unsigned short> segment;
            segment.set_data_ref((unsigned short*)volume_data->get_pixel_pointer());
            segment.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
            segment.set_dim(volume_data->_dim);
            segment.set_target_label(label);
            segment.set_min_scalar(volume_data->get_min_scalar());
            segment.set_max_scalar(volume_data->get_max_scalar());
            segment.segment_auto_threshold(ellipsoid, SegmentThreshold<unsigned short>::Otsu);

            ConnectedDomainAnalysis cd_analy;
            cd_analy.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
            cd_analy.set_dim(volume_data->_dim);
            cd_analy.set_target_label(label);
            cd_analy.set_roi(aabb._min , aabb._max);
            cd_analy.keep_major();

            break;
        }
    default:
        APPCOMMON_THROW_EXCEPTION("Unsupported data type!");
    }

    //Update to texture
    if (GPU == Configuration::instance()->get_processing_unit_type()) {
        if (aabb != AABBUI()) {
            update_aabb_i(aabb);
        }
    }
}

void OBAnnotationSegment::update_aabb_i(const AABBUI& aabb)
{
    unsigned int dim_brick[3] = {aabb._max[0] - aabb._min[0],
        aabb._max[1] - aabb._min[1],
        aabb._max[2] - aabb._min[2]};
    unsigned char* mask_updated = new unsigned char[dim_brick[0]*dim_brick[1]*dim_brick[2]];

    std::shared_ptr<ImageData> mask_data = _volume_infos->get_mask();
    unsigned char* mask_array = (unsigned char*)mask_data->get_pixel_pointer();

    const unsigned int layer_whole = mask_data->_dim[0]*mask_data->_dim[1];
    const unsigned int layer_brick = dim_brick[0]*dim_brick[1];

    for (unsigned int z = 0  ; z< dim_brick[2] ; ++z) {
        for (unsigned int y = 0 ; y < dim_brick[1] ; ++y) {
            int zz = z+aabb._min[2];
            int yy = y+aabb._min[1];
            memcpy(mask_updated + z*dim_brick[0]*dim_brick[1] + y*dim_brick[0],
                mask_array + zz*layer_whole + yy*mask_data->_dim[0] + aabb._min[0]  ,dim_brick[0]);
        }
    }

    _volume_infos->update_mask(aabb._min , aabb._max , mask_updated);
}

MED_IMG_END_NAMESPACE