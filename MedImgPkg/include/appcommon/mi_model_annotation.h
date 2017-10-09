#ifndef MED_IMG_APPCOMMON_MI_MODEL_ANNOTATION_H
#define MED_IMG_APPCOMMON_MI_MODEL_ANNOTATION_H

#include <map>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_model_interface.h"
#include "io/mi_voi.h"
#include "arithmetic/mi_volume_statistician.h"

MED_IMG_BEGIN_NAMESPACE

class AppCommon_Export ModelAnnotation : public IModel {
public:
    enum CodeID
    {
        ADD = 0, //add one annotation by drawing a circle(sphere)
        DELETE = 1, //delete one the annotation
        MODIFYING = 2, //modifying when mouse moving
        MODIFY_COMPLETED = 3, //modify completed when mouse release
        //LOAD = 4, //load from DB, different with ADD
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

protected:

private:
    std::map<std::string, ModelAnnotation::AnnotationUnit>  _annotations;
    bool _visibility;
};

MED_IMG_END_NAMESPACE
#endif