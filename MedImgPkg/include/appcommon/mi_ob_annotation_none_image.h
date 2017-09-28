#ifndef MED_IMG_APPCOMMON_MI_OB_ANNOTATION_NONE_IMAGE_H
#define MED_IMG_APPCOMMON_MI_OB_ANNOTATION_NONE_IMAGE_H

#include <memory>
#include <vector>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_observer_interface.h"

MED_IMG_BEGIN_NAMESPACE
class AppNoneImage;
class ModelAnnotation;
class AppCommon_Export OBAnnotationNoneImg : public IObserver{
public:
    OBAnnotationNoneImg();
    virtual ~OBAnnotationNoneImg();

    void set_model(std::shared_ptr<ModelAnnotation> model);
    void set_mpr_none_image(std::vector<std::shared_ptr<AppNoneImage>> app_noneimgs);

    virtual void update(int code_id = 0);

private:
    std::weak_ptr<ModelAnnotation> _model;
    std::vector<std::shared_ptr<AppNoneImage>> _app_noneimgs;
};

MED_IMG_END_NAMESPACE

#endif