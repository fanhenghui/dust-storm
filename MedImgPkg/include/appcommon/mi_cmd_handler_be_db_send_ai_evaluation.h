#ifndef MED_IMG_APPCOMMON_CMD_HANDLER_BE_DB_SEND_AI_AEVALUATION_H
#define MED_IMG_APPCOMMON_CMD_HANDLER_BE_DB_SEND_AI_AEVALUATION_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerBE_DBSendAIEvaluation : public ICommandHandler {
public:
    CmdHandlerBE_DBSendAIEvaluation(std::shared_ptr<AppController> controller);

    virtual ~CmdHandlerBE_DBSendAIEvaluation();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AppController> _controller;

};

MED_IMG_END_NAMESPACE


#endif