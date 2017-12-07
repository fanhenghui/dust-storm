#ifndef MED_IMG_PACS_RETRIEVE_RESPONSE_COMMAND_HANDLER_H
#define MED_IMG_PACS_RETRIEVE_RESPONSE_COMMAND_HANDLER_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerPACSRetrieveResponse : public ICommandHandler {
public:
    CmdHandlerPACSRetrieveResponse(std::shared_ptr<AppController> controller);
    virtual ~CmdHandlerPACSRetrieveResponse();

    virtual int handle_command(const IPCDataHeader& dataheader, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
private:
    DISALLOW_COPY_AND_ASSIGN(CmdHandlerPACSRetrieveResponse);
};

MED_IMG_END_NAMESPACE

#endif