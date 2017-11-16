#include "mi_db_operation_query_dicom.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpQueryDICOM::DBOpQueryDICOM() {

}

DBOpQueryDICOM::~DBOpQueryDICOM() {

}

int DBOpQueryDICOM::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;

    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        APPCOMMON_THROW_EXCEPTION("parse mouse message failed!");
    }

    const std::string series_id = msg.context();
    msg.Clear();
    
    std::shared_ptr<DBServerController> controller = _db_server_controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<DB> db = controller->get_db();
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy();

    DB::ImgItem item;
    if(0 != db->get_dcm_item(series_id, item) ) {
        //return to client error
    } else {
        const std::string path = item.dcm_path;
        MI_DBSERVER_LOG(MI_DEBUG) << "series: " << series_id << ". path: " << item.dcm_path;
        std::set<std::string> postfix;
        postfix.insert(".dcm");
        std::vector<std::string> files;
        FileUtil::get_all_file_recursion(path, postfix, files);
        if(files.empty()) {
            //TODO error
        } else {
            //batch read file(Don't need to use mutl-thread)
            for (size_t i = 0; i < files.size(); ++i) {
                char* buffer = nullptr;
                unsigned int size = 0;
                if(0 != FileUtil::read_raw_ext(files[i], buffer, size) ) {
                    
                } else {
                    IPCDataHeader header;
                    header.receiver = _header.receiver;
                    header.data_len = size;
                    header.msg_id = COMMAND_ID_DB_SEND_DICOM_SERIES;
                    header.msg_info2 = i==files.size()-1 ? 1:0;
                    header.msg_info3 = files.size();
                    IPCPackage* package = new IPCPackage(header, buffer);
                    if(0 != server_proxy->async_send_data(package) ){
                        delete package;
                        package = nullptr;
                        MI_DBSERVER_LOG(MI_WARNING) << "send dcm to client failed.(client disconnected)";
                        break;
                    }
                }
            }
        }
    }

    return 0;
}

MED_IMG_END_NAMESPACE