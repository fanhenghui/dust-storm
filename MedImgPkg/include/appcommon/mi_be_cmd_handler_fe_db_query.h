#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_DB_QUERY_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_DB_QUERY_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEDBQuery : public ICommandHandler {
public:
    explicit BECmdHandlerFEDBQuery(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEDBQuery();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif