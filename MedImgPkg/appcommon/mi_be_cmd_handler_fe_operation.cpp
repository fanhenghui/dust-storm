#include "mi_be_cmd_handler_fe_operation.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_operation_factory.h"

#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEOperation::BECmdHandlerFEOperation(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

BECmdHandlerFEOperation::~BECmdHandlerFEOperation() {

}

int BECmdHandlerFEOperation::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN operation cmd handler.";
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    const unsigned int op_id = ipcheader.op_id;
    std::shared_ptr<IOperation> op = OperationFactory::instance()->get_operation(op_id);
    if (op) {
        op->reset();
        op->set_data(ipcheader , buffer);
        op->set_controller(controller);
        controller->get_thread_model()->push_operation(op);
    } else {
        MI_APPCOMMON_LOG(MI_ERROR) << "cant find operation: " << op_id;
        if (nullptr != buffer) {
            delete [] buffer;
        }
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT operation cmd handler.";
    return 0;
}


MED_IMG_END_NAMESPACE
