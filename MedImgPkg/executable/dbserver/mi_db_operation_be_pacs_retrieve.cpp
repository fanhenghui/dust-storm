#include "mi_db_operation_be_pacs_retrieve.h"

#include "util/mi_ipc_server_proxy.h"

#include "io/mi_pacs_communicator.h"
#include "io/mi_protobuf.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpBEPACSRetrieve::DBOpBEPACSRetrieve() {

}

DBOpBEPACSRetrieve::~DBOpBEPACSRetrieve() {

}

int DBOpBEPACSRetrieve::execute() {    
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBEPACSRetrieve.";

    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);
    std::shared_ptr<PACSCommunicator> pacs_commu = controller->get_pacs_communicator();
    DBSERVER_CHECK_NULL_EXCEPTION(pacs_commu);

    std::vector<DcmInfo> dcm_infos;
    if (0 != pacs_commu->retrieve_all_series(dcm_infos)) {
        MI_DBSERVER_LOG(MI_ERROR) << "PACS retrive all series failed.";
        //TODO send message to notify BE
        return -1; 
    }

    //DEBUG
    {
        int id = 0;
        for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
            const std::string series_id = (*it).series_id;
            MI_DBSERVER_LOG(MI_DEBUG) << id++ << " " << series_id;
        }
    }
    

    if (dcm_infos.empty()) {
        MI_DBSERVER_LOG(MI_ERROR) << "retrieve 0 result from PACS.";
        return -1;
    }

    //send to BE
    MsgDcmInfoCollection msg_dcm_info_collection;
    for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
        const std::string series_id = (*it).series_id;
        MsgDcmInfo* msg_dcm_info = msg_dcm_info_collection.add_dcminfo();
        msg_dcm_info->set_study_id((*it).study_id);
        msg_dcm_info->set_series_id((*it).series_id);
        msg_dcm_info->set_study_date((*it).study_date);
        msg_dcm_info->set_study_time((*it).study_time);
        msg_dcm_info->set_patient_id((*it).patient_id);
        msg_dcm_info->set_patient_name((*it).patient_name);
        msg_dcm_info->set_patient_sex((*it).patient_sex);
        msg_dcm_info->set_patient_age((*it).patient_age);
        msg_dcm_info->set_patient_birth_date((*it).patient_birth_date);
        msg_dcm_info->set_modality((*it).modality);
    }

    char* buffer = nullptr;
    int buffer_size = 0;
    if (0 != protobuf_serialize(msg_dcm_info_collection, buffer, buffer_size)) {
        MI_DBSERVER_LOG(MI_ERROR) << "serialize dicom info collection message failed.";
        return -1;
    }
    msg_dcm_info_collection.Clear();
    
    IPCDataHeader header;
    header.receiver = _header.receiver;
    header.data_len = buffer_size;
    header.msg_id = COMMAND_ID_BE_DB_PACS_RETRIEVE_RESULT;
    IPCPackage* package = new IPCPackage(header, buffer);
    if (0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send PACS retrieve result to client failed.(client disconnected)";
    }
    
    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBEPACSRetrieve.";
    return 0;
}

MED_IMG_END_NAMESPACE