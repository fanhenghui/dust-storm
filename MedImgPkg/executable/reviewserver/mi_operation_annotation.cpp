#include "mi_operation_annotation.h"

#include "appcommon/mi_app_cell.h"
#include "appcommon/mi_app_common_logger.h"
#include "appcommon/mi_app_controller.h"
#include "util/mi_ipc_client_proxy.h"

#include "mi_model_annotation.h"
#include "mi_review_logger.h"
#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

OpAnnotation::OpAnnotation() {}

OpAnnotation::~OpAnnotation() {}

int OpAnnotation::execute() {
    MI_REVIEW_LOG(MI_TRACE) << "IN OpAnnotation.";
    if(_buffer == nullptr || _header._data_len < 0) {
        MI_REVIEW_LOG(MI_ERROR) << "incompleted annotation message.";
        return -1;
    }

    MsgAnnotationUnit msgAnnotation;
    if (!msgAnnotation.ParseFromArray(_buffer, _header._data_len)) {
        MI_REVIEW_LOG(MI_ERROR) << "parse annotation message failed.";
        return -1;
    }

    const unsigned int cell_id = _header._cell_id;
    std::shared_ptr<AppController> controller = _controller.lock();
    REVIEW_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<IModel> model_i = controller->get_model(MI_MODEL_ID_ANNOTATION);
    if (!model_i) {
        MI_REVIEW_LOG(MI_ERROR) << "annotation null.";
        return -1;
    }
    std::shared_ptr<ModelAnnotation> model = std::dynamic_pointer_cast<ModelAnnotation>(model_i);
    if (!model) {
        MI_REVIEW_LOG(MI_ERROR) << "error model id to acquire annotation model.";
        return -1;
    }


    int annotation_type = msgAnnotation.type();
    int annotation_status = msgAnnotation.status();
    if (annotation_status = ModelAnnotation::ADD) {

    } else if (annotation_status = ModelAnnotation::DELETE) {

    } else if (annotation_status = ModelAnnotation::MODIFYING) {
        //TODO change annotation non-image direction to prevent update non-image when rendering

    } else if (annotation_status = ModelAnnotation::MODIFY_COMPLETED) {

    }

    MI_REVIEW_LOG(MI_TRACE) << "OUT OpAnnotation.";
    return 0;
}

MED_IMG_END_NAMESPACE