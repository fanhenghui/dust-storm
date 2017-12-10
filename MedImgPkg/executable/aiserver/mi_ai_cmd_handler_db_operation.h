#ifndef MEDIMG_AI_MI_AI_CMD_HANDLER_DB_OPERATION_H
#define MEDIMG_AI_MI_AI_CMD_HANDLER_DB_OPERATION_H

#include <memory>

#include "mi_ai_server_logger.h"
#include "mi_ai_server_common.h"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class AIServerController;
class CmdHandlerAIOperating : public ICommandHandler {
public:
    explicit CmdHandlerAIOperating(std::shared_ptr<AIServerController> controller);

    virtual ~CmdHandlerAIOperating();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AIServerController> _controller;
};

MED_IMG_END_NAMESPACE

#endif