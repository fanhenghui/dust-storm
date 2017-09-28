#include "mi_model_annotation.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

ModelAnnotation::ModelAnnotation() :_visibility(true), _notify_cache(-1) {}

ModelAnnotation::~ModelAnnotation() {}

void ModelAnnotation::add_annotation(const VOISphere &voi, unsigned char label) {
    _annotations.push_back(voi);
    _intensity_infos.push_back(IntensityInfo());
    _labels.push_back(label);
    set_changed();
}

void ModelAnnotation::remove_annotation(int id) {
    if (id < _annotations.size()) {
        auto it = _annotations.begin();
        auto it2 = _intensity_infos.begin();
        auto it3 = _labels.begin();
        int i = 0;
        while (i != id) {
            ++it;
            ++it2;
            ++it3;
            ++i;
        }

    _annotations.erase(it);
    _intensity_infos.erase(it2);
    _labels.erase(it3);
    set_changed();
  }
}

void ModelAnnotation::remove_annotation_list_rear() {
    if (!_annotations.empty()) {
        _annotations.erase(--_annotations.end());
        _intensity_infos.erase(--_intensity_infos.end());
        _labels.erase(--_labels.end());
        set_changed();
    }
}

void ModelAnnotation::remove_all() {
    std::vector<VOISphere>().swap(_annotations);
    std::vector<IntensityInfo>().swap(_intensity_infos);
    std::vector<unsigned char>().swap(_labels);
    set_changed();
}

const std::vector<VOISphere> &ModelAnnotation::get_annotations() const {
    return _annotations;
}

VOISphere ModelAnnotation::get_annotation(int id) const {
    if (id < _annotations.size()) {
        return _annotations[id];
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation failed!");
    }
}

unsigned char ModelAnnotation::get_label(int id) const {
    if (id < _labels.size()) {
        return _labels[id];
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation label failed!");
    }
}

const std::vector<unsigned char> &ModelAnnotation::get_labels() const {
    return _labels;
}

void ModelAnnotation::modify_annotation_list_rear(const VOISphere &voi) {
    if (!_annotations.empty()) {
        _annotations[_annotations.size() - 1].diameter = voi.diameter;
        _annotations[_annotations.size() - 1].center = voi.center;
    }
    set_changed();
}

void ModelAnnotation::modify_diameter(int id, double diameter) {
  if (id < _annotations.size()) {
    _annotations[id].diameter = diameter;
    set_changed();
  } else {
      MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
  }
}

void ModelAnnotation::modify_center(int id, const Point3 &center) {
    if (id < _annotations.size()) {
        _annotations[id].center = center;
        set_changed();
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
    }
}

void ModelAnnotation::modify_intensity_info(int id, IntensityInfo info) {
    if (id < _intensity_infos.size()) {
        _intensity_infos[id] = info;
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
    }
}

IntensityInfo ModelAnnotation::get_intensity_info(int id) {
    if (id < _intensity_infos.size()) {
        return _intensity_infos[id];
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation intensity info failed!");
    }
}

int ModelAnnotation::get_annotation_count() const {
    return _annotations.size();
}

const std::vector<IntensityInfo> &ModelAnnotation::get_intensity_infos() const {
    return _intensity_infos;
}

bool ModelAnnotation::get_annotation_visibility() const {
     return _visibility; 
}

void ModelAnnotation::set_annotation_visibility(bool flag) {
    _visibility = flag;
}

VOISphere ModelAnnotation::get_annotation_by_label(unsigned char id) const {
    for (int i = 0; i < _labels.size(); ++i) {
        if (id == _labels[i]) {
            return _annotations[i];
        }
    }
    MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
    APPCOMMON_THROW_EXCEPTION("get annotation by mask label failed!");
}

void ModelAnnotation::notify_cache(int process_annotation_id) {
    _notify_cache = process_annotation_id;
}

int ModelAnnotation::get_notify_cache() const {
    return _notify_cache;
}

MED_IMG_END_NAMESPACE