#include "mi_operation_annotation.h"

#include "appcommon/mi_app_cell.h"
#include "appcommon/mi_app_common_logger.h"
#include "appcommon/mi_app_controller.h"
#include "util/mi_ipc_client_proxy.h"

#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

OpAnnotation::OpAnnotation() {}

OpAnnotation::~OpAnnotation() {}

int OpAnnotation::execute() {

  if (_header._data_len != 0) {
    const unsigned int cell_id = _header._cell_id;
    REVIEW_CHECK_NULL_EXCEPTION(_buffer);

    MsgAnnotation msgAnnotation;

    if (!msgAnnotation.ParseFromArray(_buffer, _header._data_len)) {
      REVIEW_THROW_EXCEPTION("parse mouse message failed!");
    }

    MI_APPCOMMON_LOG(MI_WARNING) << "Received Msg Update Annotation";
    // std::cout << "Received Msg Update Annotation" << std::endl;

    int t = msgAnnotation.type();
    const float cx = msgAnnotation.para1();
    const float cy = msgAnnotation.para2();
    const float r = msgAnnotation.para3();
  } else {
    IPCDataHeader header;
    header._msg_id = COMMAND_ID_BE_SEND_ANNOTATION;

    std::shared_ptr<AppController> controller = _controller.lock();
    REVIEW_CHECK_NULL_EXCEPTION(controller);
    MI_APPCOMMON_LOG(MI_WARNING) << "Delete FE ANNOTATION";
    controller->get_client_proxy()->async_send_message(header, nullptr);
    MI_APPCOMMON_LOG(MI_WARNING) << "OUT Delete FE ANNOTATION.";
  }
  return 0;
}

MED_IMG_END_NAMESPACE