#ifndef MEDIMG_DB_MI_CMD_DB_HANDLER_AI_OPERATING_H
#define MEDIMG_DB_MI_CMD_DB_HANDLER_AI_OPERATING_H

#include <memory>

#include "mi_db_server_logger.h"
#include "mi_db_server_common.h"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class DBServerController;
class DBCmdHandlerAIOperation : public ICommandHandler {
public:
    explicit DBCmdHandlerAIOperation(std::shared_ptr<DBServerController> controller);

    virtual ~DBCmdHandlerAIOperation();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<DBServerController> _controller;
};

MED_IMG_END_NAMESPACE

#endif