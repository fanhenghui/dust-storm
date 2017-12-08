#ifndef MED_IMG_APPCOMMON_CMD_HANDLER_BE_DB_PACS_FETCH_RESULT_H
#define MED_IMG_APPCOMMON_CMD_HANDLER_BE_DB_PACS_FETCH_RESULT_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerBE_DBPACSFetchResult : public ICommandHandler {
public:
    CmdHandlerBE_DBPACSFetchResult(std::shared_ptr<AppController> controller);
    virtual ~CmdHandlerBE_DBPACSFetchResult();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
private:
    DISALLOW_COPY_AND_ASSIGN(CmdHandlerBE_DBPACSFetchResult);
};

MED_IMG_END_NAMESPACE

#endif