#ifndef MED_IMG_APPCOMMON_MI_OB_ANNOTATION_LIST_H
#define MED_IMG_APPCOMMON_MI_OB_ANNOTATION_LIST_H

#include <memory>
#include <vector>
#include <map>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_observer_interface.h"
#include "io/mi_voi.h"

MED_IMG_BEGIN_NAMESPACE
class AppController;
class ModelAnnotation;
class AppCommon_Export OBAnnotationList : public IObserver{
public:
    OBAnnotationList();
    virtual ~OBAnnotationList();

    void set_model(std::shared_ptr<ModelAnnotation> model);
    void set_controller(std::shared_ptr<AppController> controller);

    virtual void update(int code_id = 0);

private:
    std::weak_ptr<ModelAnnotation> _model;
    std::weak_ptr<AppController> _controller;
    std::map<std::string, VOISphere> _pre_vois;
};

MED_IMG_END_NAMESPACE

#endif