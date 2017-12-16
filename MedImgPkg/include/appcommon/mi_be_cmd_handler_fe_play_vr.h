#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_PLAY_VR_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_PLAY_VR_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include "util/mi_operation_interface.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEPlayVR : public ICommandHandler {
public:
    explicit BECmdHandlerFEPlayVR(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEPlayVR();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    void logic(IPCDataHeader header);

private:
    std::weak_ptr<AppController> _controller;
    bool _playing;
};

MED_IMG_END_NAMESPACE

#endif