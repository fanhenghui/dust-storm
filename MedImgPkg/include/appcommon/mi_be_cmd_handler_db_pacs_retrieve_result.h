#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_DB_PACS_RETRIEVE_RESULT_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_DB_PACS_RETRIEVE_RESULT_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerDBPACSRetrieveResult : public ICommandHandler {
public:
    explicit BECmdHandlerDBPACSRetrieveResult(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerDBPACSRetrieveResult();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif