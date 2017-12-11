#include "mi_be_cmd_handler_fe_pacs_fetch.h"

#include <boost/algorithm/string.hpp>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_message.pb.h"
#include "io/mi_configure.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_common_util.h"
#include "mi_model_pacs.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPACSFetch::BECmdHandlerFEPACSFetch(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEPACSFetch::~BECmdHandlerFEPACSFetch() {}

int BECmdHandlerFEPACSFetch::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS fetch";
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<ModelPACS> model_pacs = AppCommonUtil::get_model_pacs(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_pacs);

    //check msg
    MsgString msg;
    if (!msg.ParseFromArray(buffer, dataheader.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse series message from FE PACS fetch failed.";
        MemShield shield(buffer);
        msg.Clear();
        return -1;
    }
    const std::string series_idx_str = msg.context();
    msg.Clear();

    std::vector<std::string> series_idx;
    boost::split(series_idx, series_idx_str, boost::is_any_of("|"));
    if (series_idx.empty()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "FE fetch PACS series index empty.";
        return -1;
    }

    bool query_more_than_0 = false;
    MsgDcmInfoCollection msg_response;
    for (auto it = series_idx.begin(); it != series_idx.end(); ++it) {
        //must check null string, because atoi("") will return 0!
        if((*it).empty()) {
            continue;
        }

        int idx = atoi(((*it).c_str()));
        DcmInfo dcm_info;
        if (0 != model_pacs->query_dicom(idx, dcm_info)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "FE fetch invalid series index: " << idx;
            continue;
        }
        query_more_than_0 = true;
        MsgDcmInfo* item = msg_response.add_dcminfo();
        item->set_study_id(dcm_info.study_id);
        item->set_series_id(dcm_info.series_id);
        item->set_study_date(dcm_info.study_date);
        item->set_study_time(dcm_info.study_time);
        item->set_patient_id(dcm_info.patient_id);
        item->set_patient_name(dcm_info.patient_name);
        item->set_patient_sex(dcm_info.patient_sex);
        item->set_patient_age(dcm_info.patient_age);
        item->set_patient_birth_date(dcm_info.patient_birth_date);
        item->set_modality(dcm_info.modality);

        MI_APPCOMMON_LOG(MI_INFO) << "FE fetch series : " << dcm_info.series_id;
    }
    if (!query_more_than_0 ) {
        msg_response.Clear();
        return -1;
    }

    const int msg_buffer_size = msg_response.ByteSize();
    char* msg_buffer = new char[msg_buffer_size];
    if (!msg_response.SerializeToArray(msg_buffer, msg_buffer_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialize FE fetch series message failed.";
        delete [] msg_buffer;
        msg_buffer = nullptr;
        msg_response.Clear();
        return -1;
    }
    msg_response.Clear();

    //send message to DBS to fetch choosed DICOM series
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_PACS_FETCH;
    header.data_len = msg_buffer_size;
    IPCPackage* package = new IPCPackage(header,msg_buffer);
    if(0 != controller->get_client_proxy_dbs()->sync_send_data(package)) {
        delete package;
        MI_APPCOMMON_LOG(MI_ERROR) << "send to DB to fetch PACS failed.";
        return -1;
    }    

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS fetch";
    return 0;
}

MED_IMG_END_NAMESPACE