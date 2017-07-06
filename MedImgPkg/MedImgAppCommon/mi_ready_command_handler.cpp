#include "mi_ready_command_handler.h"

#include "mi_app_controller.h"

#include <arpa/inet.h>

MED_IMG_BEGIN_NAMESPACE

ReadyCommandHandler::ReadyCommandHandler(std::shared_ptr<AppController> controller):_controller(controller)
{

}

ReadyCommandHandler::~ReadyCommandHandler()
{

}

int ReadyCommandHandler::handle_command(const IPCDataHeader& ipcheader , void* buffer)
{
    std::shared_ptr<AppController> controller = _controller.lock();
    if(nullptr == controller){
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    const unsigned int data_type = ipcheader._data_type;
    const unsigned int big_end = ipcheader._big_end;
    const unsigned int data_len = ipcheader._data_len;

    if(data_len != 4){
        APPCOMMON_THROW_EXCEPTION("Invalid pid length!");
    }
    if(nullptr == buffer){
        APPCOMMON_THROW_EXCEPTION("Invalid pid buffer!");
    }
    unsigned int v = ((unsigned int*)buffer)[0];
    pid_t server_pid = static_cast<pid_t>( ntohl(v) );
    controller->set_server_pid(server_pid);

    return 0;
}


MED_IMG_END_NAMESPACE
