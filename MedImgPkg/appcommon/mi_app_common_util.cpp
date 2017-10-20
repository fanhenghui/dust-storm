#include "mi_app_common_util.h"
#include "mi_app_controller.h"
#include "util/mi_model_interface.h"

#include "mi_model_annotation.h"
#include "mi_model_crosshair.h"

MED_IMG_BEGIN_NAMESPACE

std::shared_ptr<ModelAnnotation> AppCommonUtil::get_model_annotation(std::shared_ptr<AppController> controller) {
    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_ANNOTATION);
    if (nullptr == model_) {
        return nullptr;
    } else {
        std::shared_ptr<ModelAnnotation> model = std::dynamic_pointer_cast<ModelAnnotation>(model_);
        return model;
    }
}

std::shared_ptr<ModelCrosshair> AppCommonUtil::get_model_crosshair(std::shared_ptr<AppController> controller) {
    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_CROSSHAIR);
    if (nullptr == model_) {
        return nullptr;
    } else {
        std::shared_ptr<ModelCrosshair> model = std::dynamic_pointer_cast<ModelCrosshair>(model_);
        return model;
    }
}

MED_IMG_END_NAMESPACE