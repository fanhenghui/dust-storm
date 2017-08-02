#include "mi_shut_down_command_handler.h"

#include "mi_app_controller.h"

#include <arpa/inet.h>

MED_IMG_BEGIN_NAMESPACE

ShutDownCommandHandler::ShutDownCommandHandler(std::shared_ptr<AppController> controller):_controller(controller)
{

}

ShutDownCommandHandler::~ShutDownCommandHandler()
{

}

int ShutDownCommandHandler::handle_command(const IPCDataHeader& ipcheader , char* buffer)
{
    std::shared_ptr<AppController> controller = _controller.lock();
    if(nullptr == controller){
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    const unsigned int data_type = ipcheader._data_type;
    const unsigned int big_end = ipcheader._big_end;
    const unsigned int data_len = ipcheader._data_len;

    
    //TODO return shutdown ID
    const int quit_id = 2;
    return quit_id;
}


MED_IMG_END_NAMESPACE