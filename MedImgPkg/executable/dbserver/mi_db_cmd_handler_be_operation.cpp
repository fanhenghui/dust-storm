#include "mi_db_cmd_handler_be_operation.h"

#include "util/mi_operation_interface.h"
#include "util/mi_operation_factory.h"

#include "mi_db_server_controller.h"
#include "mi_db_server_thread_model.h"

MED_IMG_BEGIN_NAMESPACE

DBCmdHandlerBEOperation::DBCmdHandlerBEOperation(std::shared_ptr<DBServerController> controller):
    _controller(controller) {

}

DBCmdHandlerBEOperation::~DBCmdHandlerBEOperation() {

}

int DBCmdHandlerBEOperation::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBCmdHandlerBEOperation.";
    std::shared_ptr<DBServerController> controller = _controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    const unsigned int op_id = ipcheader.op_id;
    std::shared_ptr<IOperation> op = OperationFactory::instance()->get_operation(op_id);
    if (nullptr == op) {
        MI_DBSERVER_LOG(MI_ERROR) << "invalid DB server BE's operation: " << op_id;
        return 0;
    }
    if (op) {
        op->reset();
        op->set_data(ipcheader, buffer);
        op->set_controller(controller);
        controller->get_thread_model()->push_operation_be(op);
    } else {
        MI_DBSERVER_LOG(MI_ERROR) << "cant find operation: " << op_id;
        if (nullptr != buffer) {
            delete [] buffer;
        }
    }

    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBCmdHandlerBEOperation.";
    return 0;
}

MED_IMG_END_NAMESPACE
