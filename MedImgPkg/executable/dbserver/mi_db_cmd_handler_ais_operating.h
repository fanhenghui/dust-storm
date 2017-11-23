#ifndef MED_IMG_MI_CMD_HANDLER_DB_SERVER_AIS_OPERATING_H
#define MED_IMG_MI_CMD_HANDLER_DB_SERVER_AIS_OPERATING_H

#include <memory>

#include "mi_db_server_logger.h"
#include "mi_db_server_common.h"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class DBServerController;
class CmdHandlerDBAISOperating : public ICommandHandler {
public:
    CmdHandlerDBAISOperating(std::shared_ptr<DBServerController> controller);

    virtual ~CmdHandlerDBAISOperating();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<DBServerController> _controller;

};

MED_IMG_END_NAMESPACE

#endif