#ifndef MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_BACKTO_WORKLIST_H
#define MEDIMG_APPCOMMON_MI_BE_CMD_HANDLER_FE_BACKTO_WORKLIST_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export BECmdHandlerFEBackToWorklist : public ICommandHandler {
public:
    explicit BECmdHandlerFEBackToWorklist(std::shared_ptr<AppController> controller);
    virtual ~BECmdHandlerFEBackToWorklist();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AppController> _controller;

    boost::mutex _mutex;
    boost::condition _condition;
};

MED_IMG_END_NAMESPACE

#endif