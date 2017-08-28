#include "mi_ready_command_handler.h"

#include <iostream>

#include "util/mi_ipc_client_proxy.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"

MED_IMG_BEGIN_NAMESPACE 

ReadyCommandHandler::ReadyCommandHandler(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

ReadyCommandHandler::~ReadyCommandHandler() {}

int ReadyCommandHandler::handle_command(const IPCDataHeader &ipcheader,
                                        char *buffer) {
  std::shared_ptr<AppController> controller = _controller.lock();
  if (nullptr == controller) {
    APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
  }

  IPCDataHeader header;
  header._msg_id = COMMAND_ID_BE_READY;

  std::cout << "sending ready to FE.\n";
  controller->get_client_proxy()->async_send_message(header, nullptr);

  return 0;
}

MED_IMG_END_NAMESPACE
