#ifndef MED_IMG_MPR_PLAY_COMMAND_HANDLER_H
#define MED_IMG_MPR_PLAY_COMMAND_HANDLER_H

#include "mi_review_common.h"
#include "util/mi_ipc_common.h"
#include <memory>
#include "appcommon/mi_operation_interface.h"


MED_IMG_BEGIN_NAMESPACE

class AppController;
class MPRPlayCommandHandler : public ICommandHandler
{
public:
    MPRPlayCommandHandler(std::shared_ptr<AppController> controller);

    virtual ~MPRPlayCommandHandler();

    virtual int handle_command(const IPCDataHeader& datahaeder , char* buffer);

private:
    void logic_i(OpDataHeader& op_header, char* buffer);

private:
    std::weak_ptr<AppController> _controller;
    bool _playing;

};

MED_IMG_END_NAMESPACE

#endif