#include "mi_be_cmd_handler_fe_db_retrieve.h"

#include <vector>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_db.h"
#include "io/mi_protobuf.h"
#include "io/mi_configure.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEDBRetrieve::BECmdHandlerFEDBRetrieve(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEDBRetrieve::~BECmdHandlerFEDBRetrieve() {}

int BECmdHandlerFEDBRetrieve::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEDBRetrieve";
    
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

    MsgDcmInfoCollection msg;
    const static std::string UNKNOWN = "UNKNOWN";
    for (auto it = items.begin(); it != items.end(); ++it) {
        const std::string series_id = (*it).series_id;
        MsgDcmInfo* msg_dcm_info = msg.add_dcminfo();
        msg_dcm_info->set_study_id((*it).study_id.empty()?UNKNOWN:(*it).study_id);
        msg_dcm_info->set_series_id((*it).series_id.empty()?UNKNOWN:(*it).series_id);
        msg_dcm_info->set_patient_id((*it).patient_id.empty()?UNKNOWN:(*it).patient_id);
        msg_dcm_info->set_patient_name((*it).patient_name.empty()?UNKNOWN:(*it).patient_name);
        msg_dcm_info->set_modality((*it).modality.empty()?UNKNOWN:(*it).modality);
    }
    char* msg_buffer = nullptr;
    int msg_size = 0;
    if (0 != protobuf_serialize(msg,msg_buffer,msg_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "decode DICOM info collection failed.";
        return -1;
    }
    MemShield shield2(msg_buffer);

    IPCDataHeader header;
    header.sender = static_cast<unsigned int>(controller->get_local_pid());
    header.receiver = static_cast<unsigned int>(controller->get_server_pid());
    header.msg_id = COMMAND_ID_FE_BE_DB_RETRIEVE_RESULT;
    header.data_len = msg_size;

    controller->get_client_proxy()->sync_send_data(header, msg_buffer);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEDBRetrieve";
    return 0;
}

MED_IMG_END_NAMESPACE