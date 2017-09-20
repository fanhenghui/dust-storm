#ifndef MEDIMGAPPCOMMON_MI_HEART_BEAT_COMMAND_HANDLER_H
#define MEDIMGAPPCOMMON_MI_HEART_BEAT_COMMAND_HANDLER_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export HeartBeatCommandHandler : public ICommandHandler {
public:
  HeartBeatCommandHandler(std::shared_ptr<AppController> controller);

  virtual ~HeartBeatCommandHandler();

  virtual int handle_command(const IPCDataHeader &datahaeder, char *buffer);

private:
  std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif