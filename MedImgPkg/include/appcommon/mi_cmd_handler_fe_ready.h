#ifndef MED_IMG_APPCOMMON_MI_CMD_HANDLER_FE_READY_H
#define MED_IMG_APPCOMMON_MI_CMD_HANDLER_FE_READY_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerFEReady : public ICommandHandler {
public:
    CmdHandlerFEReady(std::shared_ptr<AppController> controller);

    virtual ~CmdHandlerFEReady();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AppController> _controller;

};

MED_IMG_END_NAMESPACE

#endif