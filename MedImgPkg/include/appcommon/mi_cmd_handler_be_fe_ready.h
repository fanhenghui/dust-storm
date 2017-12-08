#ifndef MED_IMG_APPCOMMON_CMD_HANDLER_BE_FE_READY_H
#define MED_IMG_APPCOMMON_CMD_HANDLER_BE_FE_READY_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerBE_FEReady : public ICommandHandler {
public:
    CmdHandlerBE_FEReady(std::shared_ptr<AppController> controller);

    virtual ~CmdHandlerBE_FEReady();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AppController> _controller;

};

MED_IMG_END_NAMESPACE

#endif