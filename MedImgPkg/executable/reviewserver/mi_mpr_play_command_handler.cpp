#include "mi_mpr_play_command_handler.h"

#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_thread_model.h"
#include "appcommon/mi_operation_factory.h"

#include "boost/thread.hpp"

#ifdef WIN32
#else
#include <unistd.h>
#endif

MED_IMG_BEGIN_NAMESPACE

MPRPlayCommandHandler::MPRPlayCommandHandler(std::shared_ptr<AppController> controller):
    _controller(controller), _playing(false) {

}

MPRPlayCommandHandler::~MPRPlayCommandHandler() {

}

int MPRPlayCommandHandler::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    if (_playing) {
        return 0;
    }

    std::shared_ptr<AppController> controller = _controller.lock();

    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }


    const unsigned int cell_id = ipcheader._msg_info0;
    const unsigned int op_id = ipcheader._msg_info1;
    OpDataHeader op_header;
    op_header._cell_id = 0;
    op_header._op_id = OPERATION_ID_MPR_PAGING;
    op_header._data_type = ipcheader._data_type;
    op_header._big_end = ipcheader._big_end;
    op_header._data_len = ipcheader._data_len;

    boost::thread th(boost::bind(&MPRPlayCommandHandler::logic_i , this , boost::ref(op_header) ,
                                 buffer));
    th.detach();

    return 0;
}


void MPRPlayCommandHandler::logic_i(OpDataHeader& op_header, char* buffer) {
    _playing = true;
    std::shared_ptr<AppController> controller = _controller.lock();

    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    for (int i = 0 ; i < 2000 ; ++i) {
        usleep(50000);//50 ms 的播放速度
        std::shared_ptr<IOperation> op = OperationFactory::instance()->get_operation(
                                             OPERATION_ID_MPR_PAGING);

        if (op) {

            op->set_data(op_header , buffer);
            op->set_controller(controller);
            controller->get_thread_model()->push_operation(op);
        } else {
            //TODO
            APPCOMMON_THROW_EXCEPTION("cant find operation!");
        }
    }

    _playing = false;
}

MED_IMG_END_NAMESPACE
