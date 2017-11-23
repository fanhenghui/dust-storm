#ifndef MED_IMG_APPCOMMON_MI_CMD_HANDLER_BACK_TO_WORKLIST_H
#define MED_IMG_APPCOMMON_MI_CMD_HANDLER_BACK_TO_WORKLIST_H

#include <memory>
#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerBackToWorklist : public ICommandHandler {
public:
    CmdHandlerBackToWorklist(std::shared_ptr<AppController> controller);

    virtual ~CmdHandlerBackToWorklist();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    std::weak_ptr<AppController> _controller;

    boost::mutex _mutex;
    boost::condition _condition;
};

MED_IMG_END_NAMESPACE

#endif