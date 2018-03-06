#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_PACS_GET_SERIES_LIST_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_PACS_GET_SERIES_LIST_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEPACSGetSeriesList : public ICommandHandler {
public:
    explicit BECmdHandlerFEPACSGetSeriesList(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEPACSGetSeriesList();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif