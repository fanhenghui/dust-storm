#include "mi_heart_beat_command_handler.h"

#include "util/mi_ipc_client_proxy.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

HeartBeatCommandHandler::HeartBeatCommandHandler(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

HeartBeatCommandHandler::~HeartBeatCommandHandler() {}

int HeartBeatCommandHandler::handle_command(const IPCDataHeader &ipcheader,
                                            char *buffer) {
  std::shared_ptr<AppController> controller = _controller.lock();

  if (nullptr == controller) {
    APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
  }

  IPCDataHeader header;
  header._msg_id = COMMAND_ID_BE_HEARTBEAT;

  MI_APPCOMMON_LOG(MI_INFO) << "BE reveive heart beat.";
  controller->get_client_proxy()->async_send_message(header, nullptr);

  return 0;
}

MED_IMG_END_NAMESPACE
