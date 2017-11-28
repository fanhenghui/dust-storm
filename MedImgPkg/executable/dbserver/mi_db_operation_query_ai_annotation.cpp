#include "mi_db_operation_query_ai_annotation.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "io/mi_nodule_set_parser.h"
#include "io/mi_nodule_set.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"
#include "mi_db_server_thread_model.h"
#include "mi_db_operation_request_inference.h"


MED_IMG_BEGIN_NAMESPACE

DBOpQueryAIAnnotation::DBOpQueryAIAnnotation() {

}

DBOpQueryAIAnnotation::~DBOpQueryAIAnnotation() {

}

int DBOpQueryAIAnnotation::execute() {
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
    //TODO read and check version
    if (item.annotation_ai_path.empty()) {
        //create message to request AIS inference
        //Should set client socket id to header(recv AIS result and send calcualted result to requested client) 
        MsgInferenceRequest msg;
        msg.set_series_uid(series_id);
        msg.set_dcm_path(item.dcm_path);
        msg.set_ai_anno_path(item.dcm_path+"/"+series_id+".csv");
        msg.set_client_socket_id(receiver);
        if (item.ai_intermediate_data_path.empty()) {
            msg.set_ai_im_data_path(item.dcm_path+"/"+series_id+".npy");
            msg.set_recal_im_data(true);
        } else {
            msg.set_ai_im_data_path(item.ai_intermediate_data_path);
            msg.set_recal_im_data(false);
        }
        int msg_buffer_size = msg.ByteSize();
        char* msg_buffer = new char[msg_buffer_size];
        if (msg_buffer_size != 0 && msg.SerializeToArray(msg_buffer,msg_buffer_size)){
            OpDataHeader op_header;
            op_header.data_len = msg_buffer_size;
            op_header.receiver = controller->get_ais_socket_id();
            std::shared_ptr<DBOpRequestInference> op(new DBOpRequestInference());
            op->set_data(op_header , msg_buffer);
            op->set_controller(controller);
            controller->get_thread_model()->push_operation_ais(op);
        } 
        msg.Clear();
        MI_DBSERVER_LOG(MI_INFO) << "send request to AIS.";
        //SEND_ERROR_TO_BE(server_proxy, receiver, "AI annotation file path null.");
        return 0;
    }

    NoduleSetParser parser;
    parser.set_series_id(series_id);
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    if( IO_SUCCESS != parser.load_as_csv(item.annotation_ai_path, nodule_set) ) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "load annotation file failed.");
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
        SEND_ERROR_TO_BE(server_proxy, receiver, "serialize message for AI annotation failed.");
        delete [] buffer;
        return -1;
    }
    msgAnnos.Clear();
    
    IPCDataHeader header;
    header.receiver = _header.receiver;
    header.msg_id = COMMAND_ID_DB_SEND_AI_ANNOTATION;
    header.data_len = buffer_size;
    IPCPackage* package = new IPCPackage(header,buffer); 
    if(0 != server_proxy->async_send_data(package)) {
        delete package;
        package = nullptr;
        MI_DBSERVER_LOG(MI_WARNING) << "send AI annotation to client failed.(client disconnected)";
        return -1;
    }

    return 0;
}

MED_IMG_END_NAMESPACE