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
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);
    std::shared_ptr<PACSCommunicator> pacs_commu = controller->get_pacs_communicator();
    DBSERVER_CHECK_NULL_EXCEPTION(pacs_commu);

    MsgDcmQueryKey query_key;
    if (0 != protobuf_parse(_buffer, _header.data_len, query_key)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse PACS query key message send by BE failed.";
        return -1;        
    }

    //query key
    PatientInfo patient_key;
    StudyInfo study_key;
    SeriesInfo series_key;

    patient_key.patient_id = query_key.patient_id();
    patient_key.patient_name = query_key.patient_name();
    patient_key.patient_birth_date = query_key.patient_birth_date();

    study_key.study_date = query_key.study_date();
    study_key.study_time = query_key.study_time();
    study_key.accession_no = query_key.accession_no();

    series_key.modality = query_key.modality();

    std::vector<PatientInfo> patient_infos;
    std::vector<StudyInfo> study_infos;
    std::vector<SeriesInfo> series_infos;
    if (0 != pacs_commu->query_series(patient_key, study_key, series_key, &patient_infos, &study_infos, &series_infos)) {
        MI_DBSERVER_LOG(MI_ERROR) << "PACS query series failed.";
        //TODO send message to notify BE
        return -1; 
    }

    MsgStudyWrapperCollection study_wrapper_col;
    int64_t pre_study_pk = -1;
    MsgStudyWrapper* cur_study_wrapper = nullptr;
    study_wrapper_col.set_num_study(study_infos.size());
    for (size_t i = 0; i < study_infos.size(); ++i) {
        StudyInfo& study_info = study_infos[i];
        SeriesInfo& series_info = series_infos[i];
        PatientInfo& patient_info = patient_infos[i];
        if (pre_study_pk != study_info.id) {
            cur_study_wrapper = study_wrapper_col.add_study_wrappers();

            MsgStudyInfo* msg_study_info = cur_study_wrapper->mutable_study_info();
            msg_study_info->set_patient_fk(study_info.patient_fk);
            msg_study_info->set_study_date(study_info.study_date);
            msg_study_info->set_study_time(study_info.study_time);
            msg_study_info->set_accession_no(study_info.accession_no);
            msg_study_info->set_study_desc(study_info.study_desc);
            msg_study_info->set_num_series(study_info.num_series);
            msg_study_info->set_num_instance(study_info.num_instance);

            MsgPatientInfo* msg_patient_info = cur_study_wrapper->mutable_patient_info();
            msg_patient_info->set_patient_id(patient_info.patient_id);
            msg_patient_info->set_patient_name(patient_info.patient_name);
            msg_patient_info->set_patient_birth_date(patient_info.patient_birth_date);

            MsgSeriesInfo* msg_series_info = cur_study_wrapper->add_series_infos();
            msg_series_info->set_id(series_info.id);
            msg_series_info->set_series_no(series_info.series_no);
            msg_series_info->set_modality(series_info.modality);
            msg_series_info->set_institution(series_info.institution);
            msg_series_info->set_num_instance(series_info.num_instance);

        } else {
            MsgSeriesInfo* msg_series_info = cur_study_wrapper->add_series_infos();
            msg_series_info->set_id(series_info.id);
            msg_series_info->set_series_no(series_info.series_no);
            msg_series_info->set_modality(series_info.modality);
            msg_series_info->set_institution(series_info.institution);
            msg_series_info->set_num_instance(series_info.num_instance);
        }
    }

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
    return 0;
}

MED_IMG_END_NAMESPACE