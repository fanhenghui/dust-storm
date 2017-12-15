#ifndef MED_IMG_APPCOMMON_MI_MODEL_ANNOTATION_H
#define MED_IMG_APPCOMMON_MI_MODEL_ANNOTATION_H

#include <map>
#include <set>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"
#include "io/mi_voi.h"
#include "arithmetic/mi_volume_statistician.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelAnnotation : public IModel {
public:
    enum CodeID
    {
        ADD = 0, //add one annotation by drawing a circle(sphere) or add annotations from csv file
        DELETE = 1, //delete one the annotation
        MODIFYING = 2, //modifying when mouse moving
        MODIFY_COMPLETED = 3, //modify completed when mouse release
        FOCUS = 4, //focus certain voi in each scenes
    };

    struct AnnotationUnit {
        VOISphere voi;
        unsigned char label;
        IntensityInfo intensity_info;
    };

    ModelAnnotation();
    virtual ~ModelAnnotation();

    void add_annotation(const VOISphere& voi, const std::string& id, unsigned char label);
    VOISphere get_annotation(const std::string& id) const;
    VOISphere get_annotation_by_label(unsigned char label) const;
    const std::map<std::string, ModelAnnotation::AnnotationUnit>& get_annotations() const;
    std::string get_last_annotation() const;

    unsigned char get_label(const std::string& id) const;

    void remove_annotation(const std::string& id);
    void remove_all();

    int get_annotation_count() const;

    IntensityInfo get_intensity_info(const std::string& id);
    const std::vector<IntensityInfo>& get_intensity_infos() const;
    void modify_intensity_info(const std::string& id , IntensityInfo info);

    void modify_name(const std::string& id , std::string name);
    void modify_diameter(const std::string& id , double diameter);
    void modify_center(const std::string& id , const Point3& center);

    bool get_annotation_visibility() const;
    void set_annotation_visibility(bool flag);

    void set_processing_cache(const std::vector<std::string>& ids);
    void get_processing_cache(std::vector<std::string>& ids);

    void set_probability_threshold(float thres);
    float get_probability_threshold() const;
    std::set<std::string> get_filter_annotation_ids(float probability) const;
    std::map<std::string, ModelAnnotation::AnnotationUnit> get_filter_annotations(float probability) const;
protected:

private:
    std::map<std::string, ModelAnnotation::AnnotationUnit>  _annotations;
    bool _visibility;
    float _probability_threshold;

    std::vector<std::string> _cache_ids;

    std::string _last_annotation;

private:
    DISALLOW_COPY_AND_ASSIGN(ModelAnnotation);
};

MED_IMG_END_NAMESPACE
#endif