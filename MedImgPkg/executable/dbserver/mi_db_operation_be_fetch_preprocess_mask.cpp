#include "mi_db_operation_be_fetch_preprocess_mask.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "io/mi_db.h"
#include "io/mi_protobuf.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpBEFetchPreprocessMask::DBOpBEFetchPreprocessMask() {

}

DBOpBEFetchPreprocessMask::~DBOpBEFetchPreprocessMask() {

}

int DBOpBEFetchPreprocessMask::execute() {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBEFetchPreprocessMask.";
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    MsgPreprocessRetrieveKey msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse fetch preprocess mask message send by BE failed.";
        return -1;
    }
     
    //const std::string& series_id = msg.series_uid();
    const int64_t series_pk = msg.series_pk();
    const int prep_type = msg.prep_type();
    msg.Clear();

    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    const unsigned int receiver = _header.receiver;

    std::shared_ptr<DB> db = controller->get_db();
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();

    PreprocessInfo prep_key;
    prep_key.series_fk = series_pk;
    prep_key.prep_type = prep_type;
    std::vector<PreprocessInfo> prep_infos;
    if (0 != db->query_preprocess(prep_key, &prep_infos)) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "query preprocess failed.");
        return -1;
    }
    if (prep_infos.empty()) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "query preprocess empty.");
        return -1;
    }

    if (prep_infos[0].file_path.empty()) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "preprocess file path null.");
        return -1;
    }

    char* buffer = nullptr;
    unsigned int size = 0;
    if(0 != FileUtil::read_raw_ext(prep_infos[0].file_path, buffer, size) ) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "preprocess file read failed.");
        return -1; 
    }

    IPCDataHeader header;
    header.receiver = _header.receiver;
    header.data_len = size;
    header.msg_id = COMMAND_ID_BE_DB_SEND_PREPROCESS_MASK;
    IPCPackage* package = new IPCPackage(header, buffer);
    if (0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send preprocess mask to client failed.(client disconnected)";
    }

    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBEFetchPreprocessMask.";
    return 0;
}

MED_IMG_END_NAMESPACE