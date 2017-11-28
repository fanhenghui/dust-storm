#include "mi_db_operation_receive_evaluation.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "io/mi_nodule_set_parser.h"
#include "io/mi_nodule_set.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpReceiveEvaluation::DBOpReceiveEvaluation() {

}

DBOpReceiveEvaluation::~DBOpReceiveEvaluation() {

}

int DBOpReceiveEvaluation::execute() {
    DBSERVER_CHECK_NULL_EXCEPTION(_buffer);

    std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();

    MsgEvaluationResponse msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        APPCOMMON_THROW_EXCEPTION("parse evaluation response message failed!");
    }

    //Check error
    const int status = msg.status();
    unsigned int client_socket_id = (unsigned int)msg.client_socket_id();
    if (status == -1) {
        std::string err_msg = msg.err_msg();
        SEND_ERROR_TO_BE(server_proxy,client_socket_id,err_msg);
        return -1;
    }

    //Update DB
    const std::string series_id = msg.series_uid();
    const std::string ai_anno_path = msg.ai_anno_path();
    const std::string ai_im_data_path = msg.ai_im_data_path();
    bool recal_im_data = msg.recal_im_data();

    std::shared_ptr<DB> db = controller->get_db();
    if (recal_im_data) {
        if (!ai_im_data_path.empty()){
            db->update_ai_intermediate_data(series_id, ai_im_data_path);
        } else {
            MI_DBSERVER_LOG(MI_ERROR) << "update empty AI intermediate data path.";
        }
    }
    
    if (ai_anno_path.empty()){
        SEND_ERROR_TO_BE(server_proxy,client_socket_id,"update empty AI annotation data path.");
        return -1;
    }
    
    db->update_ai_annotation(series_id, ai_anno_path);

    //load annotation and send to client
    NoduleSetParser parser;
    parser.set_series_id(series_id);
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    if( IO_SUCCESS != parser.load_as_csv(ai_anno_path, nodule_set) ) {
        SEND_ERROR_TO_BE(server_proxy, client_socket_id, "load annotation file failed.");
        return -1;
    }
    MsgAnnotationCollectionDB msgAnnos;
    msgAnnos.set_series_uid(series_id);
    const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
    for (auto it = vois.begin(); it != vois.end(); ++it) {
        const VOISphere &voi = *it;
        MsgAnnotationUnitDB* anno = msgAnnos.add_annotation();
        anno->set_x(voi.center.x);
        anno->set_y(voi.center.y);        
        anno->set_z(voi.center.z);        
        anno->set_r(voi.diameter);
        anno->set_p(voi.para0);
    }
    const int buffer_size = msgAnnos.ByteSize();
    char* buffer = new char[buffer_size];
    if (!msgAnnos.SerializeToArray(buffer, buffer_size)) {
        SEND_ERROR_TO_BE(server_proxy, client_socket_id, "serialize message for AI annotation failed.");
        delete [] buffer;
        return -1;
    }
    msgAnnos.Clear();
    IPCDataHeader header;
    header.receiver = client_socket_id;
    header.msg_id = COMMAND_ID_DB_SEND_AI_ANNOTATION;
    header.data_len = buffer_size;
    IPCPackage* package = new IPCPackage(header,buffer); 
    if(0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send AI annotation to client failed.(client disconnected)";
        return -1;
    }

    //send message end
    IPCDataHeader header_end;
    header_end.msg_id = COMMAND_ID_DB_SEND_END;
    header_end.receiver = client_socket_id;
    server_proxy->async_send_data(new IPCPackage(header_end));

    MI_DBSERVER_LOG(MI_DEBUG) << "receive AIS result and send to BE.";

    return 0;
}

MED_IMG_END_NAMESPACE