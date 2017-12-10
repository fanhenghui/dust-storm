#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_PACS_RETRIEVE_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_PACS_RETRIEVE_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEPACSRetrieve : public ICommandHandler {
public:
    explicit BECmdHandlerFEPACSRetrieve(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEPACSRetrieve();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif