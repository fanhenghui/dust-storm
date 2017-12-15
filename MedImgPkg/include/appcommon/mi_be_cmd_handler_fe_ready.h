#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_READY_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_READY_H

#include <memory>
#include <string>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEReady : public ICommandHandler {
public:
    explicit BECmdHandlerFEReady(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEReady();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

protected:  
    virtual int generate_ready_message_buffer(char*& buffer, int& buffer_size);
private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif