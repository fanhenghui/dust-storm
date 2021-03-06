#include "mi_be_operation_fe_adjust_evaluation_probability.h"

#include "arithmetic/mi_circle.h"
#include "util/mi_ipc_client_proxy.h"
#include "io/mi_protobuf.h"

#include "mi_model_annotation.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_util.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFEAdjustEvaluationProbability::BEOpFEAdjustEvaluationProbability() {}

BEOpFEAdjustEvaluationProbability::~BEOpFEAdjustEvaluationProbability() {}

int BEOpFEAdjustEvaluationProbability::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFEAdjustEvaluationProbability.";

    APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);
    MsgFloat msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse adjust annotation probability msg failed.";
        return -1;
    }

    const float probability = msg.value();
    msg.Clear();
    MI_APPCOMMON_LOG(MI_DEBUG) << "probability threshold: " << probability;

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelAnnotation> model = AppCommonUtil::get_model_annotation(controller);
    if (!model) {
        MI_APPCOMMON_LOG(MI_ERROR) << "error model id to acquire annotation model.";
        return -1;
    }

    const float pre_probability = model->get_probability_threshold();
    if (fabs(pre_probability - probability) < FLOAT_EPSILON) {
        MI_APPCOMMON_LOG(MI_DEBUG) << "probability stay the same.";
        return 0;
    }

    std::set<std::string> cur_annos = model->get_filter_annotation_ids(probability);
    std::set<std::string> pre_annos = model->get_filter_annotation_ids(pre_probability);
    model->set_probability_threshold(probability);//update
    std::vector<std::string> changed;
    if (probability > pre_probability) {
        // try delete
        for (auto it = pre_annos.begin(); it != pre_annos.end(); ++it) {
            if (cur_annos.find(*it) == cur_annos.end()) {
                changed.push_back(*it);
            }
        }
    } else {
        // try add
        for (auto it = cur_annos.begin(); it != cur_annos.end(); ++it) {
            if (pre_annos.find(*it) == pre_annos.end()) {
                changed.push_back(*it);
            }
        }
    }
    
    if(changed.empty()) {
        MI_APPCOMMON_LOG(MI_DEBUG) << "evauluation count stay the same.";
        return 0;
    }

    model->set_processing_cache(changed);
    model->set_changed();
    if (probability > pre_probability) {
        model->notify(ModelAnnotation::DELETE);
    } else {
        model->notify(ModelAnnotation::ADD);
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFEAdjustEvaluationProbability.";
    return 0;
}

MED_IMG_END_NAMESPACE