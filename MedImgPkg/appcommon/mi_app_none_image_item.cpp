#include "mi_app_none_image_item.h"

#include <sstream>

#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_annotation_calculator.h"
#include "renderalgo/mi_camera_calculator.h"

#include "mi_model_annotation.h"
#include "mi_app_common_logger.h"
#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

void NoneImgAnnotations::add_annotation(const NoneImgAnnotations::AnnotationUnit& anno) {
    _annotations.push_back(anno);
}

void NoneImgAnnotations::set_annotations(const std::vector<NoneImgAnnotations::AnnotationUnit> annotations) {
    _annotations = annotations;
}
const std::vector<NoneImgAnnotations::AnnotationUnit>& NoneImgAnnotations::get_annotations() const {
    return _annotations;
}

void NoneImgAnnotations::fill_msg(MsgNoneImgCollection* msg) const {
    MsgNoneImgAnnotations* annotations_msg = msg->mutable_annotations();
    for (size_t i = 0; i< _annotations.size(); ++i) {
        MsgAnnotationUnit* unit = annotations_msg->add_annotation();
        unit->set_type(_annotations[i].type);
        unit->set_id(_annotations[i].id);
        unit->set_status(_annotations[i].status);
        unit->set_visibility(_annotations[i].visibility);
        unit->set_para0(_annotations[i].para0);
        unit->set_para1(_annotations[i].para1);
        unit->set_para2(_annotations[i].para2);
    }
}

bool  NoneImgAnnotations::check_dirty() {
    std::shared_ptr<ModelAnnotation> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);
    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    APPCOMMON_CHECK_NULL_EXCEPTION(mpr_scene);
    int width(0),height(0);
    std::shared_ptr<CameraBase> camera = mpr_scene->get_camera();
    std::shared_ptr<OrthoCamera> ortho_camera = std::dynamic_pointer_cast<OrthoCamera>(camera);
    const std::vector<VOISphere>& vois = model->get_annotations();

    if (_pre_width == width && _pre_height == height 
        && *ortho_camera == _pre_camera && _pre_vois == vois) {
        return false;
    } else {
        return true;
    }
}

void  NoneImgAnnotations::update() {
    std::shared_ptr<ModelAnnotation> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    APPCOMMON_CHECK_NULL_EXCEPTION(mpr_scene);
    std::shared_ptr<VolumeInfos> volume_infos = mpr_scene->get_volume_infos();
    APPCOMMON_CHECK_NULL_EXCEPTION(volume_infos);
    std::shared_ptr<CameraCalculator> camera_cal = volume_infos->get_camera_calculator();
    APPCOMMON_CHECK_NULL_EXCEPTION(camera_cal);

    const std::vector<VOISphere> &vois = model->get_annotations();
    for (size_t i = 0; i < vois.size(); ++i) {
        AnnotationUnit unit;
        unit.type = 0;
        unit.id = (int)i;
        unit.status = 0;
        unit.visibility = false;
        unit.para0 = 0;
        unit.para1 = 0;
        unit.para2 = 0;
        Circle circle;
        if( AnnotationCalculator::patient_sphere_to_dc_circle(vois[i], camera_cal, mpr_scene, circle) ) {
            unit.visibility = true;
            unit.para0 = circle._center.x;
            unit.para1 = circle._center.y;
            unit.para2 = circle._radius;
        }
    }
}

void NoneImgCornerInfos::set_infos(NoneImgCornerInfos::PositionType pos, std::vector<std::pair<int, std::string>> infos) {
    _infos[pos] = infos;
}

void NoneImgCornerInfos::add_info(NoneImgCornerInfos::PositionType pos, std::pair<int, std::string> info) {
    auto it = _infos.find(pos);
    if (it == _infos.end()) {
        _infos[pos] = std::vector<std::pair<int, std::string>>(1,info);
    } else {
        _infos[pos].push_back(info);
    }
}

std::string NoneImgCornerInfos::to_string() const {
    //struct infos to character array
    //Format "LT|1:patientName|2:patientID\nLB....\nRT|....\nRB|....\n"
    typedef std::map<NoneImgCornerInfos::PositionType,std::vector<std::pair<int, std::string>>>::const_iterator const_it;
    std::stringstream ss;
    static const std::string pos_str[4] = {"LT", "LB", "RT", "RB"};
    for (const_it it = _infos.begin(); it != _infos.end(); ++it) {
        NoneImgCornerInfos::PositionType pos = it->first;
        if(!it->second.empty()) {
            ss << pos_str[static_cast<int>(pos)] << "|";
            for (size_t i = 0; i < it->second.size(); ++i) {
                ss << "|" << it->second[i].first << ":" << it->second[i].second;
            }
            ss << "\n";
            MI_APPCOMMON_LOG(MI_DEBUG) <<"corner info str: " << ss.str();
        }
    }
    return ss.str();
}

bool NoneImgCornerInfos::check_dirty() {
    if(!_init) {
        _init = true;
        return true;
    } else {
        return false;
    }
}

void NoneImgCornerInfos::update() {//TODO set corner based on config file
    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    std::shared_ptr<RayCastScene> raycast_scene = std::dynamic_pointer_cast<RayCastScene>(_scene);
    APPCOMMON_CHECK_NULL_EXCEPTION(raycast_scene);
    std::shared_ptr<VolumeInfos> volume_infos = raycast_scene->get_volume_infos();
    APPCOMMON_CHECK_NULL_EXCEPTION(volume_infos);
    std::shared_ptr<ImageDataHeader> header = volume_infos->get_data_header();

    // patient descriptor
    this->add_info(NoneImgCornerInfos::LT, std::make_pair(0, header->patient_name));
    this->add_info(NoneImgCornerInfos::LT, std::make_pair(1, header->patient_id));

    // parameters that can be tuned
    this->add_info(NoneImgCornerInfos::LB, std::make_pair(0, "Current Slice "));
    this->add_info(NoneImgCornerInfos::LB, std::make_pair(1, "W 100 L 100"));
    
    // volume structure descriptor
    std::stringstream ss;
    ss << header->columns << " " << header->rows << " " << header->slice_location.size();
    this->add_info(NoneImgCornerInfos::RT, std::make_pair(0, ss.str()));
    
    ss.str(std::string());
    ss << header->slice_thickness;
    this->add_info(NoneImgCornerInfos::RT, std::make_pair(1, ss.str()));

    // volume physical descriptor
    ss.str(std::string());
    ss << header->kvp;
    this->add_info(NoneImgCornerInfos::RB, std::make_pair(0, ss.str()));
}

void NoneImgCornerInfos::fill_msg(MsgNoneImgCollection* msg) const {
    MsgNoneImgCornerInfos* corner_msg = msg->mutable_corner_infos();
    const std::string str = this->to_string();
    if(!str.empty()) {
        corner_msg->set_infos(str);
    }
}

MED_IMG_END_NAMESPACE