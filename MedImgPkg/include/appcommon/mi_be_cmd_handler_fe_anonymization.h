#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_ANONYMIZATION_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_ANONYMIZATION_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEAnonymization : public ICommandHandler {
public:
    explicit BECmdHandlerFEAnonymization(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEAnonymization();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif