#ifndef MED_IMG_SHUTDOWN_COMMAND_HANDLER_H
#define MED_IMG_SHUTDOWN_COMMAND_HANDLER_H

#include "MedImgAppCommon/mi_app_common_export.h"
#include "MedImgUtil/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export ShutDownCommandHandler : public ICommandHandler
{
public:
    ShutDownCommandHandler(std::shared_ptr<AppController> controller);

    virtual ~ShutDownCommandHandler();

    virtual int handle_command(const IPCDataHeader& datahaeder , void* buffer);

private:
    std::weak_ptr<AppController> _controller;

};

MED_IMG_END_NAMESPACE

#endif