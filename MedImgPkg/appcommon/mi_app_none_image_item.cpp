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
#include "mi_model_crosshair.h"
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
    if (_annotations.empty()) {
        return;
    }
    
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
        // MI_APPCOMMON_LOG(MI_DEBUG) << "in annotation fill msg. id: " << _annotations[i].id << " status: " <<
        // _annotations[i].status << " vis: " << _annotations[i].visibility << " p0: " << _annotations[i].para0 <<
        // " p1: " << _annotations[i].para1 << " p2: " << _annotations[i].para2;
    }
}

bool  NoneImgAnnotations::check_dirty() {
    std::shared_ptr<ModelAnnotation> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);
    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    int width(1), height(1);
    _scene->get_display_size(width, height);
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    APPCOMMON_CHECK_NULL_EXCEPTION(mpr_scene);
    std::shared_ptr<CameraCalculator> camera_cal = mpr_scene->get_camera_calculator();
    APPCOMMON_CHECK_NULL_EXCEPTION(camera_cal);
    std::shared_ptr<CameraBase> camera = mpr_scene->get_camera();
    std::shared_ptr<OrthoCamera> ortho_camera = std::dynamic_pointer_cast<OrthoCamera>(camera);
    const std::map<std::string, ModelAnnotation::AnnotationUnit>& vois = model->get_annotations();
    typedef std::map<std::string, ModelAnnotation::AnnotationUnit>::const_iterator VOIConstIter;
    typedef std::map<std::string, NoneImgAnnotations::VOIUnit>::iterator PreVOIConstIter;

    //TODO intensity info

    _annotations.clear();
    bool voi_dirty = false;
    const bool camera_dirty = !(*ortho_camera == _pre_camera);
    const bool canvas_dirty = (width != _pre_width || height != _pre_height); 
    std::vector<VOIConstIter> unchanged_vois;
    //check add and modifying
    for (VOIConstIter it = vois.begin(); it != vois.end(); ++it) {
        const std::string& id = it->first;
        const VOISphere& voi = it->second.voi;
        auto it2 = _pre_vois.find(id);
        int status = -1;
        if(it2 == _pre_vois.end()) {
            status = 0;//add
        } else if(it2->second.voi != voi) {
            status = 2;//modifying
        }

        if(status != -1) {
            voi_dirty = true;
            AnnotationUnit unit;
            unit.type = 0;
            unit.id = id;
            unit.status = status;
            unit.visibility = false;
            unit.para0 = 0;
            unit.para1 = 0;
            unit.para2 = 0;
            Circle circle;
            if( AnnotationCalculator::patient_sphere_to_dc_circle(voi, camera_cal, mpr_scene, circle) ) {
                unit.visibility = true;
                unit.para0 = circle._center.x;
                unit.para1 = circle._center.y;
                unit.para2 = circle._radius;
            }
            this->add_annotation(unit);
        } else {
            unchanged_vois.push_back(it);
        }
    }

    //check delete
    for (auto it2 = _pre_vois.begin(); it2 != _pre_vois.end(); ++it2) {
        const std::string& id = it2->first;
        if (vois.find(id) == vois.end()) {
            voi_dirty = true;
            AnnotationUnit unit;
            unit.type = 0;
            unit.id = id;
            unit.status = 1;//delete
            unit.visibility = false;
            unit.para0 = 0;
            unit.para1 = 0;
            unit.para2 = 0;
            this->add_annotation(unit);
            //MI_APPCOMMON_LOG(MI_DEBUG) << "delete annotation: " << id;
        }
    }

    //check camera changed & canvas size changed
    if (camera_dirty || canvas_dirty) {
        //update all 2D contours
        for (auto it = unchanged_vois.begin(); it != unchanged_vois.end(); ++it) {
            const std::string& id = (*it)->first;
            const VOISphere& voi = (*it)->second.voi;
            AnnotationUnit unit;
            unit.type = 0;
            unit.id = id;
            unit.status = 2;//modifying
            unit.visibility = false;
            unit.para0 = 0;
            unit.para1 = 0;
            unit.para2 = 0;
            Circle circle;
            if( AnnotationCalculator::patient_sphere_to_dc_circle(voi, camera_cal, mpr_scene, circle) ) {
                unit.visibility = true;
                unit.para0 = circle._center.x;
                unit.para1 = circle._center.y;
                unit.para2 = circle._radius;
            }
            this->add_annotation(unit);
        }
    }

    if (camera_dirty) {
        _pre_camera = *ortho_camera;
    }
    if(canvas_dirty) {
        _pre_width = width;
        _pre_height = height;
    }
    if (voi_dirty) {
        _pre_vois.clear();
        for (auto it = vois.begin(); it != vois.end(); ++it) {
            const std::string& id = it->first;
            const VOISphere& voi = it->second.voi;
            const IntensityInfo& info = it->second.intensity_info;
            _pre_vois.insert(std::make_pair(id, VOIUnit(voi, info)));
        }
    }

    return camera_dirty || voi_dirty || canvas_dirty;
}

void  NoneImgAnnotations::update() {
    //do nothing
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
            ss << pos_str[static_cast<int>(pos)];
            for (size_t i = 0; i < it->second.size(); ++i) {
                ss << "|" << it->second[i].first << "," << it->second[i].second;
            }
            ss << "\n";
        }
    }
    //MI_APPCOMMON_LOG(MI_DEBUG) <<"corner info str: " << ss.str();
    return ss.str();
}

bool NoneImgCornerInfos::check_dirty() {
    _infos.clear();
    int dirty_items = 0;

    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    if (mpr_scene) {
        //just MPR has window level & page info
        APPCOMMON_CHECK_NULL_EXCEPTION(mpr_scene);
        std::shared_ptr<CameraCalculator> camera_cal = mpr_scene->get_camera_calculator();
        APPCOMMON_CHECK_NULL_EXCEPTION(camera_cal);
        std::shared_ptr<CameraBase> camera = _scene->get_camera();
        std::shared_ptr<OrthoCamera> ortho_camera = std::dynamic_pointer_cast<OrthoCamera>(camera);
        APPCOMMON_CHECK_NULL_EXCEPTION(ortho_camera);

        float ww(0), wl(0);
        mpr_scene->get_global_window_level(ww, wl);
        
        if(fabs(ww - _ww) > FLOAT_EPSILON || fabs(wl - _wl) > FLOAT_EPSILON) {
            ++dirty_items;
            _ww = ww;
            _wl = wl;
            std::stringstream ss;
            ss << "WW: " << _ww << " WL: " << _wl; 
            this->add_info(NoneImgCornerInfos::LB, std::make_pair(1, ss.str()));
        } 

        int max_page = 0;
        const int page = camera_cal->get_orthogonal_mpr_page(ortho_camera, max_page);
        if(page != _mpr_page) {
            ++dirty_items;
            _mpr_page = page;
            std::stringstream ss;
            ss << "Slice: " << _mpr_page << "/" << max_page;
            this->add_info(NoneImgCornerInfos::LB, std::make_pair(0, ss.str()));
            
        }
    }
    
    return (dirty_items != 0) || (!_init);
}

void NoneImgCornerInfos::update() {//TODO set corner based on config file
    if(_init) {
        return;
    }

    //update all infos 
    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    std::shared_ptr<RayCastScene> raycast_scene = std::dynamic_pointer_cast<RayCastScene>(_scene);
    APPCOMMON_CHECK_NULL_EXCEPTION(raycast_scene);
    std::shared_ptr<VolumeInfos> volume_infos = raycast_scene->get_volume_infos();
    APPCOMMON_CHECK_NULL_EXCEPTION(volume_infos);
    std::shared_ptr<ImageDataHeader> header = volume_infos->get_data_header();

    //clear pervious
    const static std::string UNKNOWN = "UNKNOWN";
    //LT
    std::string modality = UNKNOWN;
    switch(header->modality) {
    case CR:
        modality = "CR";
        break;
    case CT:
        modality = "CT";
        break;
    case MR:
        modality = "MR";
        break;
    case PT:
        modality = "PT";
        break;
    default:
        break;
    }
    this->add_info(NoneImgCornerInfos::LT, std::make_pair(0, header->manufacturer.empty() ? UNKNOWN : header->manufacturer));
    this->add_info(NoneImgCornerInfos::LT, std::make_pair(1, header->manufacturer_model_name.empty() ? UNKNOWN : header->manufacturer_model_name));
    this->add_info(NoneImgCornerInfos::LT, std::make_pair(2, modality));
    this->add_info(NoneImgCornerInfos::LT, std::make_pair(3, header->image_date.empty() ? UNKNOWN : header->image_date));

    //LB (MPR WL , MPR Slice)

    //RT

    this->add_info(NoneImgCornerInfos::RT, std::make_pair(0, header->patient_name.empty() ? UNKNOWN : header->patient_name));
    this->add_info(NoneImgCornerInfos::RT, std::make_pair(1, header->patient_id.empty() ? UNKNOWN : header->patient_id));
    this->add_info(NoneImgCornerInfos::RT, std::make_pair(2, header->patient_sex.empty() ? UNKNOWN : header->patient_sex));
    std::stringstream ss;
    ss << header->columns << " " << header->rows << " " << header->slice_location.size();
    this->add_info(NoneImgCornerInfos::RT, std::make_pair(3, ss.str()));
    
    //RB kvp and MPR thickness
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    if (mpr_scene) {
        Vector3 view_to = mpr_scene->get_camera()->get_view_direction();
        std::shared_ptr<CameraCalculator> camera_cal = mpr_scene->get_camera_calculator();
        const double left = fabs(view_to.dot_product(Vector3(1.0, 0.0, 0.0)));
        const double posterior = fabs(view_to.dot_product(Vector3(0.0, 1.0, 0.0)));
        const double head = fabs(view_to.dot_product(Vector3(0.0, 0.0, 1.0)));
        double thickness = 0.0;
        if(left > posterior && left > head) {
            thickness = volume_infos->get_volume()->_spacing[camera_cal->get_left_patient_axis_info().volume_coord / 2]; 
        } else if(posterior > left && posterior > head) {
            thickness = volume_infos->get_volume()->_spacing[camera_cal->get_posterior_patient_axis_info().volume_coord / 2]; 
        } else {
            thickness = volume_infos->get_volume()->_spacing[camera_cal->get_head_patient_axis_info().volume_coord / 2];
        }
        ss.str(std::string());
        ss << std::setprecision(3) << std::fixed << "thickness: " << thickness;
        this->add_info(NoneImgCornerInfos::RB, std::make_pair(0, ss.str()));
    }

    ss.str(std::string());
    ss << std::setprecision(0) << std::fixed << "kvp: " << header->kvp;
    this->add_info(NoneImgCornerInfos::RB, std::make_pair(1, ss.str()));

    _init = true;
}

void NoneImgCornerInfos::fill_msg(MsgNoneImgCollection* msg) const {
    MsgNoneImgCornerInfos* corner_msg = msg->mutable_corner_infos();
    const std::string str = this->to_string();
    if(!str.empty()) {
        corner_msg->set_infos(str);
    }
}

void NoneImgDirection::fill_msg(MsgNoneImgCollection* msg) const {
    MsgNoneImgDirection* direction_msg = msg->mutable_direction();
    direction_msg->set_info(_info);
}

bool NoneImgDirection::check_dirty() {
    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    std::shared_ptr<CameraBase> camera = _scene->get_camera();
    std::shared_ptr<OrthoCamera> ortho_camera = std::dynamic_pointer_cast<OrthoCamera>(camera);
    APPCOMMON_CHECK_NULL_EXCEPTION(ortho_camera);
    if (*ortho_camera == _pre_camera) {
        return false;
    } else {
        _pre_camera = *ortho_camera; 
        return true;
    }
}

void NoneImgDirection::update() {
    //TODO calculate infos
}


void NoneImgCrosshair::fill_msg(MsgNoneImgCollection* msg) const {
    MsgCrosshair* cross_hair = msg->mutable_crosshair();
    cross_hair->set_cx(_crosshair.x);
    cross_hair->set_cy(_crosshair.y);
    double a(0),b(0),c(0);
    _line0.to_func(a,b,c);
    cross_hair->set_l0_a(a);
    cross_hair->set_l0_b(b);
    cross_hair->set_l0_c(c);
    _line1.to_func(a,b,c);
    cross_hair->set_l1_a(a);
    cross_hair->set_l1_b(b);
    cross_hair->set_l1_c(c);

    if (_colors.size() == 3) {
        cross_hair->set_l0_color(_colors[0]);
        cross_hair->set_l1_color(_colors[1]);
        cross_hair->set_border_color(_colors[2]);
        _colors.clear();
    }
}

bool NoneImgCrosshair::check_dirty() {
    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    std::shared_ptr<CameraBase> camera = _scene->get_camera();
    std::shared_ptr<OrthoCamera> ortho_camera = std::dynamic_pointer_cast<OrthoCamera>(camera);
    APPCOMMON_CHECK_NULL_EXCEPTION(ortho_camera);
    std::shared_ptr<ModelCrosshair> model = _model.lock();
    Point3 pt_crosshair = model->get_cross_location_contineous_world();
    int width(-1), height(-1);
    _scene->get_display_size(width, height);
    if (_init && _pre_crosshair_w == pt_crosshair && _pre_width == width && _pre_height == height) {
        return false;
    } else {
        _pre_crosshair_w = pt_crosshair;
        _pre_width = width;
        _pre_height = height;
        return true;
    }
}

void NoneImgCrosshair::update() {
    std::shared_ptr<ModelCrosshair> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);
    APPCOMMON_CHECK_NULL_EXCEPTION(_scene);
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(_scene);
    if (mpr_scene) {
        Line2D lines_ndc[2];
        Line2D lines_dc[2];
        Point2 pt_ndc;
        Point2 pt_dc;
        RGBUnit colors[2];
        model->get_cross_line(mpr_scene, lines_ndc, pt_ndc, lines_dc, pt_dc, colors);
        if (!_init) {
            //first send message set color
            _colors.clear();
            _colors.push_back(colors[0].to_hex());
            _colors.push_back(colors[1].to_hex());
            _colors.push_back(model->get_border_color(mpr_scene).to_hex());
            _init = true;
        }
        _line0 = lines_dc[0];
        _line1 = lines_dc[1];
        _crosshair = pt_dc;

    } else {
        std::shared_ptr<VRScene> vr_scene = std::dynamic_pointer_cast<VRScene>(_scene);
        if (!vr_scene) {
            MI_APPCOMMON_LOG(MI_ERROR) << "invalid sence. not vr or mpr.";
            APPCOMMON_THROW_EXCEPTION("invalid sence. not vr or mpr.");
        }

        if (!_init) { _init = true;}

        //TODO VR scene
    }

}


MED_IMG_END_NAMESPACE