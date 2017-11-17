#include "mi_cmd_handler_recv_dbs_dicom_series.h"

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_client_proxy.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"

#include "mi_app_common_logger.h"
#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

CmdHandlerRecvDBSEndSignal::CmdHandlerRecvDBSEndSignal(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

CmdHandlerRecvDBSEndSignal::~CmdHandlerRecvDBSEndSignal() {

}

int CmdHandlerRecvDBSEndSignal::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN recveive DB server DICOM series cmd handler.";
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT recveive DB server DICOM series cmd handler.";
    return 0;
}


MED_IMG_END_NAMESPACE
