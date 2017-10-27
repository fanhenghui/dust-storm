#include "mi_cmd_handler_operating.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_operation_factory.h"

#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerOperating::CmdHandlerOperating(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

CmdHandlerOperating::~CmdHandlerOperating() {

}

int CmdHandlerOperating::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN operation cmd handler.";
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    const unsigned int cell_id = ipcheader._msg_info0;
    const unsigned int op_id = ipcheader._msg_info1;
    OpDataHeader op_header;
    op_header._cell_id = cell_id;
    op_header._op_id = op_id;
    op_header._data_type = ipcheader._data_type;
    op_header._big_end = ipcheader._big_end;
    op_header._data_len = ipcheader._data_len;

    std::shared_ptr<IOperation> op = OperationFactory::instance()->get_operation(op_id);
    if (op) {
        op->reset();
        op->set_data(op_header , buffer);
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
