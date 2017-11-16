#include "mi_db_operation_query_ai_annotation.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_server_proxy.h"

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

    DB::ImgItem item;
    if(0 != db->get_dcm_item(series_id, item) ) {
        //return to client error
    } else {
        
    }

    return 0;
}

MED_IMG_END_NAMESPACE