#include "mi_db_operation_pacs_fetch.h"    

#include "util/mi_ipc_server_proxy.h"
#include "util/mi_file_util.h"

#include "io/mi_pacs_communicator.h"
#include "appcommon/mi_message.pb.h"
#include "appcommon/mi_app_config.h"
#include "appcommon/mi_app_db.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpPACSFetch::DBOpPACSFetch() {

}

DBOpPACSFetch::~DBOpPACSFetch() {

}

int DBOpPACSFetch::execute() {
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
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse DICOM info collection message failed!";
        return -1;
    }

    const int series_num = msg.dcminfo_size();
    std::string series_id(""), study_id(""), study_dir(""), series_dir("");
    const std::string db_path = AppConfig::instance()->get_db_path();
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

        //2 fetch data from PACS
        if(0 != pacs_commu->fetch_series(series_id, "/home/wangrui22/data/cache")){
            MI_DBSERVER_LOG(MI_ERROR) << "PACS try fetch series: " << series_id << "failed.";
        }
        MI_DBSERVER_LOG(MI_DEBUG) << "PACS fetch series: " << series_id << "success.";

        //3 TODO push preprocess mask calculate(size mb will calculate then)

        //4 update DB
        DB::ImgItem img;
        img.series_id = series_id;
        img.study_id = study_id;
        img.patient_name = item.patient_name();
        img.patient_id = item.patient_id();
        img.modality = item.modality();
        img.dcm_path = series_dir;
        db->insert_dcm_item(img);

        //4 send response message back to BE
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

    return 0;
}

MED_IMG_END_NAMESPACE