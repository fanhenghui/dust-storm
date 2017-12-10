#ifndef MEDIMG_APPCOMMON_MI_DB_CMD_HANDLER_FE_HEARTBEAT_H
#define MEDIMG_APPCOMMON_MI_DB_CMD_HANDLER_FE_HEARTBEAT_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEHeartbeat : public ICommandHandler {
public:
    explicit BECmdHandlerFEHeartbeat(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEHeartbeat();

    virtual int handle_command(const IPCDataHeader &dataheader, char *buffer);

private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif