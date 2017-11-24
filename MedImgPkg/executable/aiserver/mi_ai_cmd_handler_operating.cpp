#include "mi_ai_cmd_handler_operating.h"

#include "mi_ai_server_controller.h"
#include "mi_ai_server_thread_model.h"
#include "appcommon/mi_operation_interface.h"
#include "appcommon/mi_operation_factory.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerAIOperating::CmdHandlerAIOperating(std::shared_ptr<AIServerController> controller):
    _controller(controller) {

}

CmdHandlerAIOperating::~CmdHandlerAIOperating() {

}

int CmdHandlerAIOperating::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_AISERVER_LOG(MI_TRACE) << "IN operation cmd handler.";
    std::shared_ptr<AIServerController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    const unsigned int op_id = ipcheader.msg_info1;
    OpDataHeader op_header;
    op_header.op_id = op_id;
    op_header.data_len = ipcheader.data_len;
    op_header.receiver = ipcheader.receiver;
    op_header.end_tag = ipcheader.msg_info2;
    op_header.reserved = ipcheader.msg_info3;

    std::shared_ptr<IOperation> op = OperationFactory::instance()->get_operation(op_id);
    if (nullptr == op) {
        MI_AISERVER_LOG(MI_ERROR) << "invalid AI server operation: " << op_id;
        return 0;
    }
    if (op) {
        op->reset();
        op->set_data(op_header, buffer);
        op->set_controller(controller);
        controller->get_thread_model()->push_operation(op);
    } else {
        MI_AISERVER_LOG(MI_ERROR) << "cant find operation: " << op_id;
        if (nullptr != buffer) {
            delete [] buffer;
        }
    }

    MI_AISERVER_LOG(MI_TRACE) << "OUT operation cmd handler.";
    return 0;
}


MED_IMG_END_NAMESPACE
