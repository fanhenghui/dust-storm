#include "mi_be_cmd_handler_fe_play_vr.h"

#include <memory>
#include "boost/thread.hpp"
#ifdef WIN32
#else
#include <unistd.h>
#endif

#include "util/mi_memory_shield.h"
#include "util/mi_operation_factory.h"

#include "io/mi_message.pb.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_define.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPlayVR::BECmdHandlerFEPlayVR(std::shared_ptr<AppController> controller):
    _controller(controller), _playing(false) {

}

BECmdHandlerFEPlayVR::~BECmdHandlerFEPlayVR() {

}

int BECmdHandlerFEPlayVR::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MemShield shield(buffer);
    
    if (_playing) {
        _playing = false;
        return 0;
    }

    std::shared_ptr<AppController> controller = _controller.lock();

    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    boost::thread th(boost::bind(&BECmdHandlerFEPlayVR::logic_i , this, ipcheader));
    th.detach();

    return 0;
}


void BECmdHandlerFEPlayVR::logic_i(IPCDataHeader header) {
    _playing = true;
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    for (int i = 0 ; i < 2000 ; ++i) {
        if (!_playing) {
            break;
        }
        usleep(50000);//50 ms 的播放速度
        std::shared_ptr<IOperation> op = OperationFactory::instance()->get_operation(
                                             OPERATION_ID_BE_FE_ROTATE);

        GOOGLE_PROTOBUF_VERIFY_VERSION;
        MsgRotation msg;
        msg.set_angle(5*3.1415926/180.0);
        msg.set_axis_x(0);
        msg.set_axis_y(1);
        msg.set_axis_z(0);
                                         
        char* buffer_rotation = new char[msg.ByteSize()];
        if (!msg.SerializeToArray(buffer_rotation, msg.ByteSize())) {
            printf("serialize rotation msg failed!\n");
            return;
        }
        header.data_len = msg.ByteSize();
        if (op) {
            op->reset();
            op->set_data(header , buffer_rotation);
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
