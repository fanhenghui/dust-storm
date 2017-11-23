#include "mi_db_cmd_handler_be_operating.h"

#include "mi_db_server_controller.h"
#include "mi_db_server_thread_model.h"
#include "mi_db_operation.h"
#include "appcommon/mi_operation_factory.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerDBBEOperating::CmdHandlerDBBEOperating(std::shared_ptr<DBServerController> controller):
    _controller(controller) {

}

CmdHandlerDBBEOperating::~CmdHandlerDBBEOperating() {

}

int CmdHandlerDBBEOperating::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_DBSERVER_LOG(MI_TRACE) << "IN operation cmd handler.";
    std::shared_ptr<DBServerController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    const unsigned int op_id = ipcheader.msg_info1;
    OpDataHeader op_header;
    op_header.op_id = op_id;
    op_header.end_tag = ipcheader.msg_info2;
    op_header.data_len = ipcheader.data_len;
    op_header.receiver = ipcheader.receiver;

    std::shared_ptr<IOperation> op = OperationFactory::instance()->get_operation(op_id);
    std::shared_ptr<DBOperation> op2 = std::dynamic_pointer_cast<DBOperation>(op);
    if (nullptr == op2) {
        MI_DBSERVER_LOG(MI_ERROR) << "invalid DB server operation: " << op_id;
        return 0;
    }
    if (op2) {
        op2->reset();
        op2->set_data(op_header , buffer);
        op2->set_db_server_controller(controller);
        controller->get_thread_model()->push_operation_be(op2);
    } else {
        MI_DBSERVER_LOG(MI_ERROR) << "cant find operation: " << op_id;
        if (nullptr != buffer) {
            delete [] buffer;
        }
    }

    MI_DBSERVER_LOG(MI_TRACE) << "OUT operation cmd handler.";
    return 0;
}


MED_IMG_END_NAMESPACE
