#include "mi_db_operation_query_dicom.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "appcommon/mi_app_db.h"

#include "mi_message.pb.h"
#include "mi_db_server_controller.h"


MED_IMG_BEGIN_NAMESPACE

DBOpQueryDCM::DBOpQueryDCM() {

}

DBOpQueryDCM::~DBOpQueryDCM() {

}

int DBOpQueryDCM::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;

    if (!msg.ParseFromArray(_buffer, _header._data_len)) {
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
            for (auto it = files.begin(); it != files.end(); ++it) {
                char* buffer = nullptr;
                unsigned int size = 0;
                if(0 != FileUtil::read_raw_ext(*it, buffer, size) ) {
                    
                } else {
                    IPCDataHeader header;
                    header._receiver = _receiver;
                    header._data_len = size;
                    //TODO client cmd handler
                    //header._msg_id = ;
                    //header._msg_info1 = ;
                    server_proxy->async_send_message(header, buffer);
                }
            }
        }

    }

    return 0;
}

MED_IMG_END_NAMESPACE