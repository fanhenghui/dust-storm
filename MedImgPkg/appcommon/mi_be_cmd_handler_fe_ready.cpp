#include "mi_be_cmd_handler_fe_ready.h"

#include <sstream>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_configure.h"
#include "io/mi_message.pb.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEReady::BECmdHandlerFEReady(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEReady::~BECmdHandlerFEReady() {}

int BECmdHandlerFEReady::handle_command(const IPCDataHeader& ipcheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEReady.";

    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    IPCDataHeader header;
    header.msg_id = COMMAND_ID_FE_BE_READY;
    char* msg_buffer = nullptr;
    int msg_buffer_size = 0;
    if (0 != generate_ready_message_buffer(msg_buffer,msg_buffer_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "generate ready message buffer failed.";
        return -1;
    }

    MemShield shield2(msg_buffer);
    header.data_len = msg_buffer_size;
    controller->get_client_proxy()->sync_send_data(header, msg_buffer);

    MI_APPCOMMON_LOG(MI_INFO) << "sending ready to FE.";
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEReady";
    return 0;
}

int BECmdHandlerFEReady::generate_ready_message_buffer(char*& buffer, int& buffer_size) {
    const float probability_threshold = Configure::instance()->get_evaluation_probability_threshold();
    MsgFloat msg;
    msg.set_value(probability_threshold);
    buffer_size = msg.ByteSize();
    if (buffer_size <= 0) {
        msg.Clear();
        return -1;
    }
    buffer = new char[buffer_size];
    if (!msg.SerializeToArray(buffer, buffer_size)) {
        msg.Clear();
        delete [] buffer;
        buffer = nullptr;
        return -1;
    } else {

        return 0;
    }
}

MED_IMG_END_NAMESPACE
