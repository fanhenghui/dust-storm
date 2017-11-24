#include "mi_db_operation_query_preprocess_mask.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpQueryPreprocessMask::DBOpQueryPreprocessMask() {

}

DBOpQueryPreprocessMask::~DBOpQueryPreprocessMask() {

}

int DBOpQueryPreprocessMask::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        APPCOMMON_THROW_EXCEPTION("parse series message failed!");
    }
    const std::string series_id = msg.context();
    msg.Clear();
    
    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<DB> db = controller->get_db();
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();

    const unsigned int receiver = _header.receiver;
    DB::ImgItem item;
    if(0 != db->get_dcm_item(series_id, item) ) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "DICOM series item not existed.");
        return -1;
    }

    if (item.preprocess_mask_path.empty()) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "DICOM preprocessing mask path null.");
        return -1;
    }
    //MI_DBSERVER_LOG(MI_DEBUG) << "preprocess mask path: " << item.preprocess_mask_path;
    char* buffer = nullptr;
    unsigned int size = 0;
    if(0 != FileUtil::read_raw_ext(item.preprocess_mask_path, buffer, size) ) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "preprocess mask file not existed.");
        return -1; 
    }

    IPCDataHeader header;
    header.receiver = _header.receiver;
    header.data_len = size;
    header.msg_id = COMMAND_ID_DB_SEND_PREPROCESS_MASK;
    IPCPackage* package = new IPCPackage(header, buffer);
    if (0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send dcm to client failed.(client disconnected)";
    }

    return 0;
}

MED_IMG_END_NAMESPACE