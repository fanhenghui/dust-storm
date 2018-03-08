#include "mi_be_cmd_handler_fe_back_to_worklist.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"
#include "util/mi_operation_interface.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_util.h"
#include "mi_model_anonymization.h"
#include "mi_model_pacs_cache.h"

MED_IMG_BEGIN_NAMESPACE

class OpResetController: public IOperation {
public:
    OpResetController(boost::condition* condition=nullptr):_condition(condition) {
        
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

        //keep PACS/Anonymization model, and remove other models
        std::shared_ptr<ModelAnonymization> model_anonymization = AppCommonUtil::get_model_anonymization(controller);
        std::shared_ptr<ModelPACSCache> model_pacs_cache = AppCommonUtil::get_model_pacs_cache(controller);
        controller->remove_all_models();
        if (model_anonymization) {
            controller->add_model(MODEL_ID_ANONYMIZATION, model_anonymization);
            controller->add_model(MODEL_ID_PACS_CACHE, model_pacs_cache);
        }

        if (_condition) {
            _condition->notify_one();
        }

        return 0;
    }

    CREATE_EXTENDS_OP(OpResetController)
private:
    boost::condition* _condition;
};

BECmdHandlerFEBackToWorklist::BECmdHandlerFEBackToWorklist(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEBackToWorklist::~BECmdHandlerFEBackToWorklist() {}

int BECmdHandlerFEBackToWorklist::handle_command(const IPCDataHeader& ipcheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEBackToWorklist.";

    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    //trigger a sync operation to release cells/models/volume
    std::shared_ptr<IOperation> op(new OpResetController(&_condition));
    op->set_controller(controller);
    controller->get_thread_model()->push_operation_fe(op);

    //wait realse operation executed
    boost::mutex::scoped_lock locker(_mutex);
    _condition.wait(_mutex);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEBackToWorklist.";
    return 0;
}

MED_IMG_END_NAMESPACE
