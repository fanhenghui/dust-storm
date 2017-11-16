#include "mi_db_operation_query_ai_annotation.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

#include "io/mi_nodule_set_parser.h"
#include "io/mi_nodule_set.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"


MED_IMG_BEGIN_NAMESPACE

DBOpQueryAIAnnotation::DBOpQueryAIAnnotation() {

}

DBOpQueryAIAnnotation::~DBOpQueryAIAnnotation() {

}

int DBOpQueryAIAnnotation::execute() {
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

    const unsigned int receiver = _header.receiver;
    DB::ImgItem item;
    if(0 != db->get_dcm_item(series_id, item) ) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "DICOM series item not existed.");
        return -1;
    }
    if (item.annotation_ai_path.empty()) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "AI annotation file path null.");
        return -1;
    }

    NoduleSetParser parser;
    parser.set_series_id(item.series_id);
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    if( IO_SUCCESS != parser.load_as_csv(item.annotation_ai_path, nodule_set) ) {
        SEND_ERROR_TO_BE(server_proxy, receiver, "load annotation file failed.");
        return -1;
    }

    MsgAnnotationCollectionDB msgAnnos;
    msgAnnos.set_series_uid(item.series_id);

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