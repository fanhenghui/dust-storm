#ifndef MED_IMG_REVIEWSERVER_MI_MODEL_ANNOTATION_H
#define MED_IMG_REVIEWSERVER_MI_MODEL_ANNOTATION_H

#include "mi_review_common.h"
#include "util/mi_model_interface.h"
#include "io/mi_voi.h"
#include "arithmetic/mi_volume_statistician.h"

MED_IMG_BEGIN_NAMESPACE

class ModelAnnotation : public IModel {
public:
    enum CodeID
    {
        ADD = 0, //add one annotation by drawing a circle(sphere)
        DELETE = 1, //delete one the annotation
        MODIFYING = 2, //modifying when mouse moving
        MODIFY_COMPLETED = 3, //modify completed when mouse release
        //LOAD = 4, //load from DB, different with ADD
    };

    ModelAnnotation();
    virtual ~ModelAnnotation();

    void add_annotation(const VOISphere& voi, unsigned char label);
    VOISphere get_annotation(int id) const;
    VOISphere get_annotation_by_label(unsigned char label) const;
    const std::vector<VOISphere>& get_annotations() const;

    unsigned char get_label(int id) const;
    const std::vector<unsigned char>& get_labels() const;


    void remove_annotation_list_rear();
    void remove_annotation(int id);
    void remove_all();

    int get_annotation_count() const;

    IntensityInfo get_intensity_info(int id);
    const std::vector<IntensityInfo>& get_intensity_infos() const;
    void modify_intensity_info(int id , IntensityInfo info);


    void modify_name(int id , std::string name);
    void modify_diameter(int id , double diameter);
    void modify_center(int id , const Point3& center);
    void modify_annotation_list_rear(const VOISphere& voi);

    bool get_annotation_visibility() const;
    void set_annotation_visibility(bool flag);

protected:

private:
    std::vector<VOISphere>       _annotations;
    std::vector<unsigned char>   _labels;
    std::vector<IntensityInfo>   _intensity_infos;
    bool _visibility;
};

MED_IMG_END_NAMESPACE
#endif