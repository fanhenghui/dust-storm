#ifndef MED_IMG_APPCOMMON_MI_CMD_HANDLER_SHUTDOWN_H
#define MED_IMG_APPCOMMON_MI_CMD_HANDLER_SHUTDOWN_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerShutdown : public ICommandHandler {
public:
    CmdHandlerShutdown(std::shared_ptr<AppController> controller);

    virtual ~CmdHandlerShutdown();

    virtual int handle_command(const IPCDataHeader& datahaeder , char* buffer);

private:
    std::weak_ptr<AppController> _controller;

};

MED_IMG_END_NAMESPACE

#endif