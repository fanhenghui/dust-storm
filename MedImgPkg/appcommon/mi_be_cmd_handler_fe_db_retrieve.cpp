#include "mi_be_cmd_handler_fe_db_retrieve.h"

#include <vector>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_db.h"
#include "io/mi_message.pb.h"
#include "io/mi_configure.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEDBRetrieve::BECmdHandlerFEDBRetrieve(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEDBRetrieve::~BECmdHandlerFEDBRetrieve() {}

int BECmdHandlerFEDBRetrieve::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler search worklist";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //query series in remote DB
    std::string db_ip_port,db_user,db_pwd,db_name;
    Configure::instance()->get_db_info(db_ip_port, db_user, db_pwd, db_name);
    DB db;
    if( 0 != db.connect(db_user, db_ip_port, db_pwd, db_name)) {
        MI_APPCOMMON_LOG(MI_FATAL) << "connect DB failed.";
        return -1;
    }

    std::vector<DB::ImgItem> items;
    if (0 != db.get_all_dcm_items(items)) {
        MI_APPCOMMON_LOG(MI_FATAL) << "get all dcm items from DB failed.";
        return -1;
    }

    MsgWorklist worklist;
    const static std::string UNKNOWN = "UNKNOWN";
    for (size_t i = 0; i < items.size() ; ++i) {
        MsgWorklistItem* item = worklist.add_items();
        if (items[i].patient_id.empty()) {
            item->set_patient_id(UNKNOWN);
        } else {
            item->set_patient_id(items[i].patient_id);
        }

        if (items[i].patient_name.empty()) {
            item->set_patient_name(UNKNOWN);
        } else {
            item->set_patient_name(items[i].patient_name);
        }

        if (items[i].series_id.empty()) {
            item->set_series_uid(UNKNOWN);
        } else {
            item->set_series_uid(items[i].series_id);
        }
        
        if (items[i].modality.empty()) {
            item->set_imaging_modality(UNKNOWN);
        } else {
            item->set_imaging_modality(items[i].modality);
        }      
    }

    int size = worklist.ByteSize();
    char* data = new char[size];
    bool res = worklist.SerializeToArray(data, size);
    worklist.Clear();

    IPCDataHeader header;
    header.sender = static_cast<unsigned int>(controller->get_local_pid());
    header.receiver = static_cast<unsigned int>(controller->get_server_pid());
    header.msg_id = COMMAND_ID_FE_BE_DB_RETRIEVE_RESULT;

    if (!res) {
        size = 0;
        free(data);
        data = nullptr;
    }

    header.data_len = size;
    controller->get_client_proxy()->sync_send_data(header, data);
    if (data != nullptr) {
        delete [] data;
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler search worklist";
    return 0;
}

MED_IMG_END_NAMESPACE