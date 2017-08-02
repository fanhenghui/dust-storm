#ifndef MED_IMG_READY_COMMAND_HANDLER_H
#define MED_IMG_READY_COMMAND_HANDLER_H

#include "MedImgAppCommon/mi_app_common_export.h"
#include "MedImgUtil/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export ReadyCommandHandler : public ICommandHandler
{
public:
    ReadyCommandHandler(std::shared_ptr<AppController> controller);

    virtual ~ReadyCommandHandler();

    virtual int handle_command(const IPCDataHeader& datahaeder , char* buffer);

private:
    std::weak_ptr<AppController> _controller;

};

MED_IMG_END_NAMESPACE

#endif