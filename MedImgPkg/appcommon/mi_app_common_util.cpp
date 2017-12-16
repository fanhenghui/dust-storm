#include "mi_app_common_util.h"
#include "mi_app_controller.h"
#include "util/mi_model_interface.h"

#include "mi_model_annotation.h"
#include "mi_model_crosshair.h"
#include "mi_model_dbs_status.h"
#include "mi_model_pacs.h"
#include "mi_model_anonymization.h"

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

std::shared_ptr<ModelDBSStatus> AppCommonUtil::get_model_dbs_status(std::shared_ptr<AppController> controller) {
    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_DBS_STATUS);
    if (nullptr == model_) {
        return nullptr;
    } else {
        std::shared_ptr<ModelDBSStatus> model = std::dynamic_pointer_cast<ModelDBSStatus>(model_);
        return model;
    }
}

std::shared_ptr<ModelPACS> AppCommonUtil::get_model_pacs(std::shared_ptr<AppController> controller) {
    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_PACS);
    if (nullptr == model_) {
        return nullptr;
    } else {
        std::shared_ptr<ModelPACS> model = std::dynamic_pointer_cast<ModelPACS>(model_);
        return model;
    }
}

std::shared_ptr<ModelAnonymization> AppCommonUtil::get_model_anonymization(std::shared_ptr<AppController> controller) {
    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_ANONYMIZATION);
    if (nullptr == model_) {
        return nullptr;
    } else {
        std::shared_ptr<ModelAnonymization> model = std::dynamic_pointer_cast<ModelAnonymization>(model_);
        return model;
    }
}

MED_IMG_END_NAMESPACE