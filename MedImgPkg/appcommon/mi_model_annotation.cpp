#include "mi_model_annotation.h"
#include "io/mi_configure.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

ModelAnnotation::ModelAnnotation() :_visibility(true),
_probability_threshold(Configure::instance()->get_evaluation_probability_threshold()) {}

ModelAnnotation::~ModelAnnotation() {}

void ModelAnnotation::add_annotation(const VOISphere &voi, const std::string& id, unsigned char label) {
    AnnotationUnit unit;
    unit.voi = voi;
    unit.label = label;
    unit.intensity_info = IntensityInfo();
    _annotations[id] = unit;
    set_changed();
    _last_annotation = id;
}

void ModelAnnotation::remove_annotation(const std::string& id) {
    auto it = _annotations.find(id);
    if (it != _annotations.end()) {
        _annotations.erase(it);
    }
    if (_annotations.empty()) {
        _last_annotation.clear();
    }
    set_changed();
}

void ModelAnnotation::remove_all() {
    _annotations.clear();
    set_changed();
}

const std::map<std::string, ModelAnnotation::AnnotationUnit>& ModelAnnotation::get_annotations() const {
    return _annotations;
}

VOISphere ModelAnnotation::get_annotation(const std::string& id) const {
    auto it = _annotations.find(id);
    if (it != _annotations.end()) {
        return it->second.voi;
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation failed!");
    }
}

std::string ModelAnnotation::get_last_annotation() const {
    if (_annotations.empty() || _last_annotation.empty()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "get last annotation failed. empty annotations.";
        APPCOMMON_THROW_EXCEPTION("get last annotation failed. empty annotations.");
    }

    return _last_annotation;
}

unsigned char ModelAnnotation::get_label(const std::string& id) const {
    auto it = _annotations.find(id);
    if (it != _annotations.end()) {
        return it->second.label;
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation failed!");
    }
}

void ModelAnnotation::modify_diameter(const std::string& id, double diameter) {
    auto it = _annotations.find(id);
    if (it != _annotations.end()) {
        it->second.voi.diameter = diameter;
        set_changed();
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation failed!");
    }
}

void ModelAnnotation::modify_center(const std::string& id, const Point3 &center) {
    auto it = _annotations.find(id);
    if (it != _annotations.end()) {
        it->second.voi.center = center;
        set_changed();
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation failed!");
    }
}

void ModelAnnotation::modify_intensity_info(const std::string& id, IntensityInfo info) {
    auto it = _annotations.find(id);
    if (it != _annotations.end()) {
        it->second.intensity_info = info;
        //set_changed();
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation failed!");
    }
}

IntensityInfo ModelAnnotation::get_intensity_info(const std::string& id) {
    auto it = _annotations.find(id);
    if (it != _annotations.end()) {
        return it->second.intensity_info;
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid annotation id: " << id;
        APPCOMMON_THROW_EXCEPTION("get annotation failed!");
    }
}

int ModelAnnotation::get_annotation_count() const {
    return _annotations.size();
}

bool ModelAnnotation::get_annotation_visibility() const {
     return _visibility; 
}

void ModelAnnotation::set_annotation_visibility(bool flag) {
    _visibility = flag;
}

VOISphere ModelAnnotation::get_annotation_by_label(unsigned char label) const {
    for (auto it = _annotations.begin(); it != _annotations.end(); ++it) {
        if (it->second.label == label) {
            return it->second.voi;
        }
    }
    MI_APPCOMMON_LOG(MI_ERROR) << "can't find label: " << label << " in annotation model.";
    APPCOMMON_THROW_EXCEPTION("get annotation by mask label failed!");
}

void ModelAnnotation::set_processing_cache(const std::vector<std::string>& ids) {
    _cache_ids = ids;
}

void ModelAnnotation::get_processing_cache(std::vector<std::string>& ids) {
    ids = _cache_ids;
}

void ModelAnnotation::set_probability_threshold(float thres) {
    _probability_threshold = thres;
}

float ModelAnnotation::get_probability_threshold() const {
    return _probability_threshold;
}

std::set<std::string> ModelAnnotation::get_filter_annotation_ids(float probability) const {
    std::set<std::string> match_annos;
    for (std::map<std::string, ModelAnnotation::AnnotationUnit>::const_iterator it = _annotations.begin(); 
        it != _annotations.end(); ++it) {
        if (it->second.voi.probability >= probability) {
            match_annos.insert(it->first);
        }
    }
    return match_annos;
}

std::map<std::string, ModelAnnotation::AnnotationUnit> ModelAnnotation::get_filter_annotations(float probability) const {
    std::map<std::string, ModelAnnotation::AnnotationUnit> filter_res;
    for (std::map<std::string, ModelAnnotation::AnnotationUnit>::const_iterator it = _annotations.begin(); 
        it != _annotations.end(); ++it) {
        if (it->second.voi.probability >= probability) {
            filter_res[it->first] = it->second;
        }
    }

    return filter_res;
}

MED_IMG_END_NAMESPACE