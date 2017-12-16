#include "mi_db_operation_be_pacs_fetch.h"    

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

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpBEPACSFetch::DBOpBEPACSFetch() {

}

DBOpBEPACSFetch::~DBOpBEPACSFetch() {

}

int DBOpBEPACSFetch::execute() {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBEPACSFetch.";
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);
    
    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);
    std::shared_ptr<PACSCommunicator> pacs_commu = controller->get_pacs_communicator();
    DBSERVER_CHECK_NULL_EXCEPTION(pacs_commu);
    std::shared_ptr<DB> db = controller->get_db();
    DBSERVER_CHECK_NULL_EXCEPTION(db);

    MsgDcmInfoCollection msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse PACS fetch message send by BE failed.";
        return -1;
    }

    const int series_num = msg.dcminfo_size();
    std::string series_id(""), study_id(""), study_dir(""), series_dir("");
    const std::string db_path = Configure::instance()->get_db_path();
    for (int i = 0; i < series_num; ++i) {
        MsgDcmInfo item = msg.dcminfo(i);
        study_id = item.study_id();
        series_id = item.series_id();

        //1 create direction
        study_dir = db_path + "/" + study_id;
        if (0 != FileUtil::check_direction(study_dir)) {
            if (0 != FileUtil::make_direction(study_dir)) {
                MI_DBSERVER_LOG(MI_ERROR) << "create study direction: " << study_dir << " failed.";
                continue;
            }
        } 
        series_dir = study_dir + "/" + series_id;
        if (0 != FileUtil::check_direction(series_dir)) {
            if (0 != FileUtil::make_direction(series_dir)) {
                MI_DBSERVER_LOG(MI_ERROR) << "create series direction: " << series_dir << " failed.";
                continue;
            }
        }

        MI_DBSERVER_LOG(MI_INFO) << "PACS fetching series : " << series_id << " >>>>>>";
        //2 fetch data from PACS
        if(0 != pacs_commu->fetch_series(series_id, series_dir)){
            MI_DBSERVER_LOG(MI_ERROR) << "PACS try fetch series: " << series_id << " failed.";
            continue;
        }
        MI_DBSERVER_LOG(MI_INFO) << "PACS fetch series: " << series_id << " success.";

        //3 preprocess mask calculate(size mb will calculate then)
        // TODO 暂时使用方案2（简单版本：串行计算），方案1会比较好
        const std::string preprocess_mask_path = series_dir + "/" + series_id + ".rle";
        float dicoms_size_mb = 0;
        if(0 != preprocess_i(series_dir, preprocess_mask_path, dicoms_size_mb)) {
            MI_DBSERVER_LOG(MI_ERROR) << "preprocess PACS fetched series: " << series_id << " failed.";
            continue;
        }
        MI_DBSERVER_LOG(MI_INFO) << "preprocess PACS fetched series: " << series_id << " success.";

        //4 update DB
        DB::ImgItem img;
        img.series_id = series_id;
        img.study_id = study_id;
        img.patient_name = item.patient_name();
        img.patient_id = item.patient_id();
        img.modality = item.modality();
        img.dcm_path = series_dir;
        img.preprocess_mask_path = preprocess_mask_path;
        img.size_mb = dicoms_size_mb;
        db->insert_dcm_item(img);

        //5 send response message back to BE
        MsgString msg_response;
        msg_response.set_context(series_id);
        const int buffer_size = msg_response.ByteSize();
        char* buffer_response = new char[buffer_size];
        if (!msg_response.SerializeToArray(buffer_response, buffer_size)) {
            MI_DBSERVER_LOG(MI_ERROR) << "DB parse PACS fetch response message failed.";
            msg_response.Clear();
            delete [] buffer_response;
            continue;
        }
        msg_response.Clear();
        IPCDataHeader header;
        header.receiver = _header.receiver;
        header.data_len = buffer_size;
        header.msg_id = COMMAND_ID_BE_DB_PACS_FETCH_RESULT;
        IPCPackage* package = new IPCPackage(header, buffer_response);
        if (0 != server_proxy->async_send_data(package)) {
            delete package;
            package = nullptr;
            MI_DBSERVER_LOG(MI_WARNING) << "DB send PACS fetch response message failed.";
        }
    }

    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBEPACSFetch.";
    return 0;
}

int DBOpBEPACSFetch::preprocess_i(const std::string& series_dir, const std::string& preprocess_mask_path, float& dicoms_size_mb) {
    std::vector<std::string> files;
    std::set<std::string> dcm_postfix;
    dcm_postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(series_dir, dcm_postfix, files);
    dicoms_size_mb = FileUtil::get_size_mb(files);

    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> volume_data;
    DICOMLoader loader;
    IOStatus status = loader.load_series(files, volume_data, data_header);
    if(status != IO_SUCCESS) {
        MI_DBSERVER_LOG(MI_ERROR) << "preprocess load DICOM root " << series_dir << " failed.";
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