#include "mi_cmd_handler_be_fe_back_to_worklist.h"

#include <iostream>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "mi_operation_interface.h"
#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_app_thread_model.h"

MED_IMG_BEGIN_NAMESPACE

class OpResetController: public IOperation {
public:
    OpResetController(boost::condition* condition):_condition(condition) {
        
    }

    virtual ~OpResetController() {

    }

    virtual int execute() {
        std::shared_ptr<AppController> controller = get_controller<AppController>();
        if (nullptr == controller) {
            APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
            if (_condition) {
                _condition->notify_one();
            }
            return -1;
        }

        //trigger a sync operation to lock cmd handler
        //clear volume infos, cells , models
        controller->set_volume_infos(nullptr);
        controller->remove_all_cells();
        controller->remove_all_models();

        if (_condition) {
            _condition->notify_one();
        }

        return 0;
    }

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpResetController>(new OpResetController(nullptr));
    }
private:
    boost::condition* _condition;
    DISALLOW_COPY_AND_ASSIGN(OpResetController);
};

CmdHandlerBE_FEBackToWorklist::CmdHandlerBE_FEBackToWorklist(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerBE_FEBackToWorklist::~CmdHandlerBE_FEBackToWorklist() {}

int CmdHandlerBE_FEBackToWorklist::handle_command(const IPCDataHeader& ipcheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN back to worklist cmd handler.";
    MemShield shield(buffer);

    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    //trigger a sync operation to release cells/models/volume
    std::shared_ptr<IOperation> op(new OpResetController(&_condition));
    op->set_controller(controller);
    controller->get_thread_model()->push_operation(op);

    //wait realse operation executed
    boost::mutex::scoped_lock locker(_mutex);
    _condition.wait(_mutex);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT back to worklist cmd handler.";
    return 0;
}

MED_IMG_END_NAMESPACE
