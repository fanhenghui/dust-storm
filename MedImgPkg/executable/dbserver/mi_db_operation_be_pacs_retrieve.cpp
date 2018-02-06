#include "mi_db_operation_be_pacs_retrieve.h"    

#include "arithmetic/mi_connected_domain_analysis.h"
#include "arithmetic/mi_segment_threshold.h"
#include "arithmetic/mi_morphology.h"
#include "arithmetic/mi_ct_table_removal.h"
#include "arithmetic/mi_run_length_operator.h"

#include "util/mi_ipc_server_proxy.h"
#include "util/mi_file_util.h"

#include "io/mi_pacs_communicator.h"
#include "io/mi_protobuf.h"
#include "io/mi_configure.h"
#include "io/mi_db.h"
#include "io/mi_dicom_info.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpBEPACSRetrieve::DBOpBEPACSRetrieve() {

}

DBOpBEPACSRetrieve::~DBOpBEPACSRetrieve() {

}

int DBOpBEPACSRetrieve::execute() {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBEPACSRetrieve.";
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);
    
    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);
    std::shared_ptr<PACSCommunicator> pacs_commu = controller->get_pacs_communicator();
    DBSERVER_CHECK_NULL_EXCEPTION(pacs_commu);
    std::shared_ptr<DB> db = controller->get_db();
    DBSERVER_CHECK_NULL_EXCEPTION(db);

    MsgDcmPACSRetrieveKey retrieve_key;
    
    if (0 != protobuf_parse(_buffer, _header.data_len, retrieve_key)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse PACS retrieve message send by BE failed.";
        return -1;
    }

    if (retrieve_key.series_uid_size() != retrieve_key.study_uid_size()) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse PACS retrieve message send by BE failed 2.";
        return -1;
    }

    const int series_num = retrieve_key.series_uid_size();

    std::string study_dir(""), series_dir("");
    const std::string db_path = Configure::instance()->get_db_path();
    for (int i = 0; i < series_num; ++i) {
        const std::string& series_uid = retrieve_key.series_uid(i);
        const std::string& study_uid = retrieve_key.study_uid(i);
        //1 create direction
        study_dir = db_path + "/" + study_uid;
        if (0 != FileUtil::check_direction(study_dir)) {
            if (0 != FileUtil::make_direction(study_dir)) {
                MI_DBSERVER_LOG(MI_ERROR) << "create study direction: " << study_dir << " failed.";
                continue;
            }
        } 
        series_dir = study_dir + "/" + series_uid;
        if (0 != FileUtil::check_direction(series_dir)) {
            if (0 != FileUtil::make_direction(series_dir)) {
                MI_DBSERVER_LOG(MI_ERROR) << "create series direction: " << series_dir << " failed.";
                continue;
            }
        }

        MI_DBSERVER_LOG(MI_INFO) << "PACS retrieve series : " << series_uid << " >>>>>>";
        
        //------------------------------------------//
        // query dicom infos from PACS
        //------------------------------------------//
        SeriesInfo series_key;
        series_key.series_uid = series_uid;
        StudyInfo study_key;
        study_key.study_uid = study_uid;
        std::vector<PatientInfo> patient_infos;
        std::vector<StudyInfo> study_infos;
        std::vector<SeriesInfo> series_infos;
        if (0 != pacs_commu->query_series(PatientInfo(), study_key, series_key, &patient_infos, &study_infos, &series_infos)) {
            MI_DBSERVER_LOG(MI_ERROR) << "PACS try query series: " << series_uid << " failed.";
            continue;
        }
        
        if (patient_infos.empty()) {
            MI_DBSERVER_LOG(MI_ERROR) << "PACS try query series: " << series_uid << " empty.";
            continue;
        }

        //------------------------------------------//
        // retrieve data from PACS
        //------------------------------------------//
        std::vector<InstanceInfo> instances;
        if(0 != pacs_commu->retrieve_series(series_uid, series_dir, &instances)){
            MI_DBSERVER_LOG(MI_ERROR) << "PACS try retrieve series: " << series_uid << " failed.";
            continue;
        }
        if (instances.empty()) {
            MI_DBSERVER_LOG(MI_ERROR) << "PACS try retrieve series: " << series_uid << " empty.";
            continue;
        }
        MI_DBSERVER_LOG(MI_INFO) << "PACS retrieve series: " << series_uid << " success.";

        //------------------------------------------//
        // insert retrieved series data to DB
        //------------------------------------------//
        if(0 != db->insert_series(patient_infos[0], study_infos[0], series_infos[0], instances)) {
            MI_DBSERVER_LOG(MI_ERROR) << "insert retroeved series: " << series_uid << " to db failed.";
            continue;
        }

        //------------------------------------------//
        // preprocess mask calculate
        // TODO 这里仅仅调用了一个临时的简单mask分割，后续需要发送消息到AIS调用AI的接口（mask分割加ai计算预处理）
        //------------------------------------------//
        if (instances.size() > VOLUME_SLICE_LIMIT) {
            const std::string prep_mask_path = series_dir + "/" + series_uid + ".rle";
            if(0 != preprocess(instances, prep_mask_path)) {
                MI_DBSERVER_LOG(MI_ERROR) << "preprocess PACS retrieved series: " << series_uid << " failed.";
                continue;
            }
            
            int64_t file_size = 0;
            if (0 != FileUtil::get_file_size(prep_mask_path, file_size)) {
                MI_DBSERVER_LOG(MI_ERROR) << "get retrieved series " << series_uid << "'s preprocess file failed.";
                continue;
            }

            PreprocessInfo prep_info;
            prep_info.series_fk = series_infos[0].id;
            prep_info.prep_type = INIT_SEGMENT_MASK;
            prep_info.version = "0.0.0";
            prep_info.file_path = prep_mask_path;
            prep_info.file_size = file_size;
            db->insert_preprocess(prep_info);

            MI_DBSERVER_LOG(MI_INFO) << "preprocess PACS retrieved series: " << series_uid << " success.";
        }

        //------------------------------------------//
        //  send response message back to BE
        //------------------------------------------//
        MsgString msg_response;
        msg_response.set_context(series_uid);
        int buffer_size = 0;
        char* buffer_response = nullptr;
        if (0 != protobuf_serialize(msg_response, buffer_response, buffer_size)) {
            MI_DBSERVER_LOG(MI_ERROR) << "DB parse PACS retrieved response message failed.";
            continue;
        }
        msg_response.Clear();
        IPCDataHeader header;
        header.receiver = _header.receiver;
        header.data_len = buffer_size;
        header.msg_id = COMMAND_ID_BE_DB_PACS_RETRIEVE_RESULT;
        IPCPackage* package = new IPCPackage(header, buffer_response);
        if (0 != server_proxy->async_send_data(package)) {
            delete package;
            package = nullptr;
            MI_DBSERVER_LOG(MI_WARNING) << "DB send PACS retrieve response message failed.";
        }
    }

    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBEPACSRetrieve.";
    return 0;
}

int DBOpBEPACSRetrieve::preprocess(const std::vector<InstanceInfo>& instances, const std::string& preprocess_mask_path) {
    std::vector<std::string> files(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        files[i] = instances[i].file_path;
    }
    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> volume_data;
    DICOMLoader loader;
    IOStatus status = loader.load_series(files, volume_data, data_header);
    if(status != IO_SUCCESS) {
        MI_DBSERVER_LOG(MI_ERROR) << "preprocess load DICOM failed.";
        return -1;
    }
    MI_DBSERVER_LOG(MI_DEBUG) << "preprocess load DICOM " << data_header->series_uid << " done.";
    MI_DBSERVER_LOG(MI_DEBUG) << "preprocess DICOM dim " << volume_data->_dim[0] << " " << volume_data->_dim[1] << " " << volume_data->_dim[2];
    const unsigned int dim[3] = {volume_data->_dim[0] , volume_data->_dim[1] , volume_data->_dim[2]};
    const unsigned int volume_size = dim[0]*dim[1]*dim[2];
    std::unique_ptr<unsigned char[]> mask_(new unsigned char[volume_size]);
    unsigned char* mask = mask_.get();
    memset(mask, 0, volume_size);

    if (volume_data->_data_type == USHORT) {
        CTTableRemoval<unsigned short> removal;
        removal.set_data_ref((unsigned short*)(volume_data->get_pixel_pointer()));
        removal.set_dim(dim);
        removal.set_mask_ref(mask);
        removal.set_target_label(1);
        removal.set_min_scalar(volume_data->get_min_scalar());
        removal.set_max_scalar(volume_data->get_max_scalar());
        removal.set_image_orientation(volume_data->_image_orientation);
        removal.set_intercept(volume_data->_intercept);
        removal.set_slope(volume_data->_slope);
        removal.remove();      
    } else {
        CTTableRemoval<short> removal;
        removal.set_data_ref((short*)(volume_data->get_pixel_pointer()));
        removal.set_dim(dim);
        removal.set_mask_ref(mask);
        removal.set_target_label(1);
        removal.set_min_scalar(volume_data->get_min_scalar());
        removal.set_max_scalar(volume_data->get_max_scalar());
        removal.set_image_orientation(volume_data->_image_orientation);
        removal.set_intercept(volume_data->_intercept);
        removal.set_slope(volume_data->_slope);
        removal.remove();
    }

    //DEBUG
    // {
    //     std::stringstream ss;
    //     ss << "/home/wangrui22/data/" << data_header->series_uid << "|mask.raw";
    //     FileUtil::write_raw(ss.str(), mask, volume_size);
    // }

    std::vector<unsigned int> res = RunLengthOperator::encode(mask, volume_size);
    if (0 != FileUtil::write_raw(preprocess_mask_path, res.data(), res.size()*sizeof(unsigned int))) {
        MI_DBSERVER_LOG(MI_ERROR) << "preprocess series:  " << data_header->series_uid << " failed.";
        return -1;
    } else {
        MI_DBSERVER_LOG(MI_DEBUG) << "preprocess series:  " << data_header->series_uid << " success.";
        return 0;
    }
}
MED_IMG_END_NAMESPACE