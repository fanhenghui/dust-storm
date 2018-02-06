#include "mi_be_cmd_handler_fe_db_query.h"

#include <vector>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_db.h"
#include "io/mi_protobuf.h"
#include "io/mi_configure.h"
#include "io/mi_dicom_info.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEDBQuery::BECmdHandlerFEDBQuery(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEDBQuery::~BECmdHandlerFEDBQuery() {}

int BECmdHandlerFEDBQuery::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEDBQuery";
    
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    MsgDcmQueryKey query_key;
    if (0 != protobuf_parse(buffer, dataheader.data_len, query_key)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "encode DB query key failed."; 
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


    //query in remote DB
    std::string db_ip_port,db_user,db_pwd,db_name;
    Configure::instance()->get_db_info(db_ip_port, db_user, db_pwd, db_name);
    DB db;
    if( 0 != db.connect(db_user, db_ip_port, db_pwd, db_name)) {
        MI_APPCOMMON_LOG(MI_FATAL) << "connect DB failed.";
        return -1;
    }

    std::vector<PatientInfo> patient_infos;
    std::vector<StudyInfo> study_infos;
    std::vector<SeriesInfo> series_infos;
    if (0 != db.query_series(patient_key, study_key, series_key, &patient_infos, &study_infos, &series_infos)) {
        MI_APPCOMMON_LOG(MI_FATAL) << "query dcm series from DB failed.";
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

    char* msg_buffer = nullptr;
    int msg_size = 0;
    if (0 != protobuf_serialize(study_wrapper_col,msg_buffer,msg_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "decode DICOM info collection failed.";
        return -1;
    }
    MemShield shield2(msg_buffer);

    IPCDataHeader header;
    header.sender = static_cast<unsigned int>(controller->get_local_pid());
    header.receiver = static_cast<unsigned int>(controller->get_server_pid());
    header.msg_id = COMMAND_ID_FE_BE_DB_QUERY_RESULT;
    header.data_len = msg_size;

    controller->get_client_proxy()->sync_send_data(header, msg_buffer);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEDBQuery";
    return 0;
}

MED_IMG_END_NAMESPACE