#include "mi_be_cmd_handler_fe_anonymization.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"
#include "io/mi_protobuf.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_model_anonymization.h"
#include "mi_app_common_util.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEAnonymization::BECmdHandlerFEAnonymization(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEAnonymization::~BECmdHandlerFEAnonymization() {}

int BECmdHandlerFEAnonymization::handle_command(const IPCDataHeader &header, char *buffer) {
    MemShield shield(buffer);
    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    
    MsgFlag msg;
    if (0 != protobuf_decode<MsgFlag>(buffer, header.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse anonymazitation flag failed.";
        return -1;
    }

    const bool flag = msg.flag();
    std::shared_ptr<ModelAnonymization> model = AppCommonUtil::get_model_anonymization(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model);

    if (model->get_anonymization_flag() != flag) {
        model->set_anonymization_flag(flag);
        model->set_changed();
        model->notify();
    }

    return 0;
}

MED_IMG_END_NAMESPACE
