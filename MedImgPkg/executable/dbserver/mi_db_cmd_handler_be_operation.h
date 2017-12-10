#ifndef MEDIMG_DB_MI_DB_CMD_HANDLER_BE_OPERATION_H
#define MEDIMG_DB_MI_DB_CMD_HANDLER_BE_OPERATION_H

#include <memory>

#include "mi_db_server_logger.h"
#include "mi_db_server_common.h"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class DBServerController;
class DBCmdHandlerBEOperation : public ICommandHandler {
public:
    explicit DBCmdHandlerBEOperation(std::shared_ptr<DBServerController> controller);

    virtual ~DBCmdHandlerBEOperation();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<DBServerController> _controller;
};

MED_IMG_END_NAMESPACE

#endif