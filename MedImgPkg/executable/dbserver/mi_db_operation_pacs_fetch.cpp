#include "mi_db_operation_pacs_fetch.h"

#include <boost/algorithm/string.hpp>     

#include "util/mi_ipc_server_proxy.h"
#include "io/mi_pacs_communicator.h"
#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpPACSFetch::DBOpPACSFetch() {

}

DBOpPACSFetch::~DBOpPACSFetch() {

}

int DBOpPACSFetch::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    MsgString msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        DBSERVER_THROW_EXCEPTION("parse series message failed!");
    }
    const std::string series_id_str = msg.context();
    msg.Clear();
    
    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);
    std::shared_ptr<PACSCommunicator> pacs_commu = controller->get_pacs_communicator();
    DBSERVER_CHECK_NULL_EXCEPTION(pacs_commu);

    std::vector<std::string> series_ids;
    boost::split(series_ids, series_id_str, boost::is_any_of('|'));
    if (series_ids.empty()) {
        MI_DBSERVER_LOG(MI_ERROR) << "PACS try fetch empty series id array.";
        return -1;
    }


    //TODO set DB direction need study&series uid
    MsgString msg_response;
    for (auto it=series_ids.begin(); it != series_ids.end(); ++it) {
        const std::string& series_id = *it;
        
        if(0 != pacs_commu->fetch_series(series_id, "/home/wangrui22/data/cache")){
            MI_DBSERVER_LOG(MI_ERROR) << "PACS try fetch series: " << series_id << "failed.";
        }
        MI_DBSERVER_LOG(MI_DEBUG) << "PACS fetch series: " << series_id << "success.";

        //send success signal back
        msg_response.set_context(series_id);
        const int buffer_size = msg_response.ByteSize();
        char* buffer_response = new char[buffer_size];
        if (!msg_response.SerializeToArray(buffer_response, buffer_size)) {
            MI_DBSERVER_LOG(MI_ERROR) << "DB parse PACS fetch response message failed.";
            continue;
        } else {
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
    }


    return 0;
}

MED_IMG_END_NAMESPACE