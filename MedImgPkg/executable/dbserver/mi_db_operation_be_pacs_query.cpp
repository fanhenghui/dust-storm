#include "mi_db_operation_be_pacs_query.h"

#include "util/mi_ipc_server_proxy.h"

#include "io/mi_pacs_communicator.h"
#include "io/mi_protobuf.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpBEPACSQuery::DBOpBEPACSQuery() {

}

DBOpBEPACSQuery::~DBOpBEPACSQuery() {

}

int DBOpBEPACSQuery::execute() {    
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBEPACSQuery.";

    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);
    std::shared_ptr<PACSCommunicator> pacs_commu = controller->get_pacs_communicator();
    DBSERVER_CHECK_NULL_EXCEPTION(pacs_commu);

    //query key
    PatientInfo patient_key;
    StudyInfo study_key;
    SeriesInfo series_key;

    if (_buffer) {
        MsgDcmQueryKey query_key;
        if (0 != protobuf_parse(_buffer, _header.data_len, query_key)) {
            MI_DBSERVER_LOG(MI_ERROR) << "parse PACS query key message send by BE failed.";
            return -1;        
        }
        patient_key.patient_id = query_key.patient_id();
        patient_key.patient_name = query_key.patient_name();
        patient_key.patient_birth_date = query_key.patient_birth_date();

        study_key.study_date = query_key.study_date();
        study_key.study_time = query_key.study_time();
        study_key.accession_no = query_key.accession_no();

        series_key.modality = query_key.modality();

        //for debug
        MI_DBSERVER_LOG(MI_DEBUG) << "dcm query key: " << 
        "patient_id: " << patient_key.patient_id << std::endl << 
        "patient_name: " << patient_key.patient_name << std::endl << 
        "patient_birth_date: " << patient_key.patient_birth_date << std::endl << 
        "study_date: " << study_key.study_date << std::endl << 
        "study_time: " << study_key.study_time << std::endl << 
        "accession_no: " << study_key.accession_no << std::endl << 
        "modality: " << series_key.modality;
    } else {
        //for debug
        MI_DBSERVER_LOG(MI_WARNING) << "null query key buffer.";
    }

    std::vector<PatientInfo> patient_infos;
    std::vector<StudyInfo> study_infos;
    std::vector<SeriesInfo> series_infos;
    if (0 != pacs_commu->query_series(patient_key, study_key, series_key, &patient_infos, &study_infos, &series_infos)) {
        MI_DBSERVER_LOG(MI_ERROR) << "PACS query series failed.";
        //TODO send message to notify BE
        return -1; 
    }

    //for debug
    for (size_t i = 0; i < series_infos.size(); ++i) {  
        MI_DBSERVER_LOG(MI_DEBUG) << series_infos[i].series_uid;
    }
    
    std::vector<StudyInfo*> cp_study_info;
    std::map<std::string, size_t> cp_study_map;
    std::map<size_t, std::vector<SeriesInfo*>> cp_series_info;
    std::map<size_t, PatientInfo*> cp_patient_info;
    for (size_t i = 0; i < study_infos.size(); ++i) {

        size_t study_map_idx = 0;
        auto it = cp_study_map.find(study_infos[i].study_uid);
        if (it == cp_study_map.end()) {
            cp_study_info.push_back(&(study_infos[i]));
            cp_study_map.insert(std::make_pair(study_infos[i].study_uid, cp_study_info.size()-1));
            study_map_idx = cp_study_info.size()-1;
        } else {
            study_map_idx = it->second;
        }

        if (cp_patient_info.find(study_map_idx) == cp_patient_info.end()) {
            cp_patient_info.insert(std::make_pair(study_map_idx, &(patient_infos[i])));
        }
        
        auto it1 = cp_series_info.find(study_map_idx);
        if (it1 == cp_series_info.end()) {
            cp_series_info.insert(std::make_pair(study_map_idx, 
                std::vector<SeriesInfo*>(1, &(series_infos[i]))));
        } else {
            it1->second.push_back(&(series_infos[i]));
        }
    }

    MsgStudyWrapperCollection study_wrapper_col;
    MsgStudyWrapper* cur_study_wrapper = nullptr;
    study_wrapper_col.set_num_study(cp_study_info.size());
    for (size_t i = 0; i < cp_study_info.size(); ++i) {

        StudyInfo& study_info = *(cp_study_info[i]);
        PatientInfo& patient_info = *(cp_patient_info[i]);

        cur_study_wrapper = study_wrapper_col.add_study_wrappers();

        MsgStudyInfo* msg_study_info = cur_study_wrapper->mutable_study_info();
        msg_study_info->set_patient_fk(study_info.patient_fk);
        msg_study_info->set_study_date(study_info.study_date);
        msg_study_info->set_study_time(study_info.study_time);
        msg_study_info->set_accession_no(study_info.accession_no);
        msg_study_info->set_study_desc(study_info.study_desc);
        msg_study_info->set_study_uid(study_info.study_uid);
        msg_study_info->set_num_series(study_info.num_series);
        msg_study_info->set_num_instance(study_info.num_instance);

        MsgPatientInfo* msg_patient_info = cur_study_wrapper->mutable_patient_info();
        msg_patient_info->set_patient_id(patient_info.patient_id);
        msg_patient_info->set_patient_name(patient_info.patient_name);
        msg_patient_info->set_patient_sex(patient_info.patient_sex);
        msg_patient_info->set_patient_birth_date(patient_info.patient_birth_date);

        std::vector<SeriesInfo*> series_infos_in_study = cp_series_info[i];
        for (size_t j = 0; j < series_infos_in_study.size(); ++j) {
            SeriesInfo& series_info = *(series_infos_in_study[j]);
            MsgSeriesInfo* msg_series_info = cur_study_wrapper->add_series_infos();
            msg_series_info->set_id(series_info.id);
            msg_series_info->set_series_no(series_info.series_no);
            msg_series_info->set_modality(series_info.modality);
            msg_series_info->set_institution(series_info.institution);
            msg_series_info->set_num_instance(series_info.num_instance);
            msg_series_info->set_series_desc(series_info.series_desc);
            msg_series_info->set_series_uid(series_info.series_uid);
        }
    }

    //---------------------------------//
    //这里感觉是protobuf的bug，如果为空结果，会序列化失败，因此加了下面的代码
    if (series_infos.size() == 0) {
        study_wrapper_col.set_num_study(-1);    
    }
    //---------------------------------//

    char* buffer = nullptr;
    int buffer_size = 0;    
    if (0 != protobuf_serialize(study_wrapper_col, buffer, buffer_size)) {
        MI_DBSERVER_LOG(MI_ERROR) << "serialize dicom info collection message failed.";
        return -1;
    }
    study_wrapper_col.Clear();
    
    IPCDataHeader header;
    header.receiver = _header.receiver;
    header.data_len = buffer_size;
    header.msg_id = COMMAND_ID_BE_DB_PACS_QUERY_RESULT;
    IPCPackage* package = new IPCPackage(header, buffer);
    if (0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send PACS query result to client failed.(client disconnected)";
    }
    
    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBEPACSQuery.";

    MI_DBSERVER_LOG(MI_DEBUG) << "db op be pacs query done.";
    return 0;
}

MED_IMG_END_NAMESPACE