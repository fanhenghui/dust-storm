#ifndef MED_IMG_APPCOMMON_CMD_HANDLER_BE_FE_PACS_FETCH_H
#define MED_IMG_APPCOMMON_CMD_HANDLER_BE_FE_PACS_FETCH_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerBE_FEPACSFetch : public ICommandHandler {
public:
    CmdHandlerBE_FEPACSFetch(std::shared_ptr<AppController> controller);
    virtual ~CmdHandlerBE_FEPACSFetch();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
private:
    DISALLOW_COPY_AND_ASSIGN(CmdHandlerBE_FEPACSFetch);
};

MED_IMG_END_NAMESPACE

#endif