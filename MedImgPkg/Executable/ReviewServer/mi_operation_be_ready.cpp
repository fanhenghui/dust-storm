#include "mi_operation_be_ready.h"
#include "MedImgAppCommon/mi_app_controller.h"
#include "MedImgUtil/mi_ipc_client_proxy.h"

MED_IMG_BEGIN_NAMESPACE

OpBEReady::OpBEReady() {}

OpBEReady::~OpBEReady() {}

int OpBEReady::execute() {
  std::shared_ptr<AppController> controller(_controller.lock());
  REVIEW_CHECK_NULL_EXCEPTION(controller);

  std::shared_ptr<IPCClientProxy> client_proxy = controller->get_client_proxy();

  IPCDataHeader header;
  header._msg_id = COMMAND_ID_BE_READY;

  client_proxy->async_send_message(header, nullptr);

  return 0;
}

MED_IMG_END_NAMESPACE