#ifndef MED_IMG_APPCOMMON_MI_CMD_HANDLER_RECV_DB_SERVER_PREPCODESS_MASK_H
#define MED_IMG_APPCOMMON_MI_CMD_HANDLER_RECV_DB_SERVER_PREPCODESS_MASK_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerBE_DBSendPreprocessMask : public ICommandHandler {
public:
    CmdHandlerBE_DBSendPreprocessMask(std::shared_ptr<AppController> controller);

    virtual ~CmdHandlerBE_DBSendPreprocessMask();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AppController> _controller;
};

MED_IMG_END_NAMESPACE

#endif