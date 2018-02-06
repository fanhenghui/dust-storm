#include "mi_db_operation_be_fetch_dicom.h"

#include <time.h>

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "io/mi_db.h"
#include "io/mi_protobuf.h"

#include "appcommon/mi_app_common_define.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpBEFetchDICOM::DBOpBEFetchDICOM() {

}

DBOpBEFetchDICOM::~DBOpBEFetchDICOM() {
    
}

int DBOpBEFetchDICOM::execute() {
    MI_DBSERVER_LOG(MI_TRACE) << "IN DBOpBEFetchDICOM.";
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);
    clock_t _start = clock();

    MsgDcmDBRetrieveKey msg;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
        MI_DBSERVER_LOG(MI_ERROR) << "parse fetch DICOM message send by BE failed.";
        return -1;
    }

    const int64_t series_pk = msg.series_pk();
    msg.Clear();
    
    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<DB> db = controller->get_db();
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();

    const unsigned int receiver = _header.receiver;
    
    std::vector<std::string> instance_files;
    if(0 != db->query_series_instance(series_pk, &instance_files)) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "query instance failed.");
        return -1;
    }

    if(instance_files.empty()) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "query instance empty.");
        return -1;
    }

    //batch read file(Don't need to use mutl-thread)
    for (size_t i = 0; i < instance_files.size(); ++i) {
        char* buffer = nullptr;
        unsigned int size = 0;
        if(0 != FileUtil::read_raw_ext(instance_files[i], buffer, size) ) {
            SEND_ERROR_TO_BE(server_proxy, receiver, "read DICOM file failed.");
            return -1;
        }

        IPCDataHeader header;
        header.receiver = receiver;
        header.data_len = size;
        header.msg_id = COMMAND_ID_BE_DB_SEND_DICOM_SERIES;
        header.reserved0 = i==instance_files.size()-1 ? 1:0;
        header.reserved1 = instance_files.size();
        IPCPackage* package = new IPCPackage(header, buffer);
        if(0 != server_proxy->async_send_data(package)){
            delete package;
            package = nullptr;
            MI_DBSERVER_LOG(MI_WARNING) << "send dcm to client failed.(client disconnected)";
            break;
        }
    }
    clock_t _end = clock();
    MI_DBSERVER_LOG(MI_INFO) << "success send {series:" << series_pk << ", slice:" << instance_files.size() << ", cost:" << double(_end-_start)/CLOCKS_PER_SEC << "s}.";

    MI_DBSERVER_LOG(MI_TRACE) << "OUT DBOpBEFetchDICOM.";
    return 0;
}

MED_IMG_END_NAMESPACE