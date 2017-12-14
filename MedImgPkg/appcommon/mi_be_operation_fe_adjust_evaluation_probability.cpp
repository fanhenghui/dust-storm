#include "mi_be_operation_fe_annotation.h"

#include "arithmetic/mi_circle.h"
#include "util/mi_ipc_client_proxy.h"
#include "io/mi_message.pb.h"

#include "renderalgo/mi_mask_label_store.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_annotation_calculator.h"
#include "renderalgo/mi_volume_infos.h"

#include "mi_model_annotation.h"
#include "mi_model_crosshair.h"
#include "mi_app_cell.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_none_image_item.h"
#include "mi_app_none_image.h"

MED_IMG_BEGIN_NAMESPACE

BNOpFEAdjustEvaluationProbability::BNOpFEAdjustEvaluationProbability() {}

BNOpFEAdjustEvaluationProbability::~BNOpFEAdjustEvaluationProbability() {}

int BNOpFEAdjustEvaluationProbability::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BNOpFEAdjustEvaluationProbability.";
    if (_buffer == nullptr || _header.data_len < 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "incompleted annotation message.";
        return -1;
    }

    MsgNumber msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse adjust annotation probability msg failed.";
        return -1;
    }
    const float probability = msg.value();
    msg.Clear();

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelAnnotation> model = AppCommonUtil::get_model_annotation(controller);
    if (!model) {
        MI_APPCOMMON_LOG(MI_ERROR) << "error model id to acquire annotation model.";
        return -1;
    }

    std::set<std::string> cur_annos = model->get_filter_annotations(probability);
    model->set_prabobility_threshold(probability);
    model->set_processing_cache(cur_annos);
    model->set_changed();
    model->notify(ModelAnnotation::PROBABILITY);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BNOpFEAdjustEvaluationProbability.";
    return 0;
}

MED_IMG_END_NAMESPACE