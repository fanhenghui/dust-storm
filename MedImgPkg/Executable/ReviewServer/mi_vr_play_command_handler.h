#ifndef MED_IMG_VR_PLAY_COMMAND_HANDLER_H
#define MED_IMG_VR_PLAY_COMMAND_HANDLER_H

#include "MedImgAppCommon/mi_operation_interface.h"
#include "MedImgUtil/mi_ipc_common.h"
#include "mi_review_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class VRPlayCommandHandler : public ICommandHandler {
public:
  VRPlayCommandHandler(std::shared_ptr<AppController> controller);

  virtual ~VRPlayCommandHandler();

  virtual int handle_command(const IPCDataHeader &datahaeder, char *buffer);

private:
  void logic_i(OpDataHeader &op_header, char *buffer);

private:
  std::weak_ptr<AppController> _controller;
  bool _playing;
};

MED_IMG_END_NAMESPACE

#endif