#include "mi_be_cmd_handler_fe_db_get_series_list.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_protobuf.h"
#include "io/mi_db.h"
#include "io/mi_configure.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_model_pacs_cache.h"
#include "mi_app_common_util.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEDBGetSeriesList::BECmdHandlerFEDBGetSeriesList(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEDBGetSeriesList::~BECmdHandlerFEDBGetSeriesList() {}

int BECmdHandlerFEDBGetSeriesList::handle_command(const IPCDataHeader &dataheader, char *buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEDBGetSeriesList";

    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    MemShield shield(buffer);

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    MsgInt64 msg_study_pk;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg_study_pk)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "encode study idx failed."; 
        return -1;
    }
    const int64_t study_pk = msg_study_pk.value();
    msg_study_pk.Clear();

    //query in remote DB
    std::string db_ip_port,db_user,db_pwd,db_name;
    Configure::instance()->get_db_info(db_ip_port, db_user, db_pwd, db_name);
    DB db;
    if( 0 != db.connect(db_user, db_ip_port, db_pwd, db_name)) {
        MI_APPCOMMON_LOG(MI_FATAL) << "connect DB failed.";
        return -1;
    }

    std::vector<SeriesInfo> series_infos;
    SeriesInfo empty_series_key;
    if (0 != db.query_series(study_pk, empty_series_key, &series_infos)) {
        MI_APPCOMMON_LOG(MI_FATAL) << "query dcm series from DB failed.";
        return -1;
    }

    if (!series_infos.empty()) {
        MsgStudyWrapper msg_res;
        for (size_t i = 0; i < series_infos.size(); ++i) {
            MsgSeriesInfo* series_info = msg_res.add_series_infos();
            series_info->set_id(series_infos[i].id);
            series_info->set_series_no(series_infos[i].series_no);
            series_info->set_modality(series_infos[i].modality);
            series_info->set_series_desc(series_infos[i].series_desc);
            series_info->set_institution(series_infos[i].institution);
            series_info->set_num_instance(series_infos[i].num_instance);
        }

        char* buffer_res = nullptr;
        int buffer_size = 0;
        if (0 != protobuf_serialize(msg_res, buffer_res, buffer_size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "serialized series list msg buffer failed.";
            return -1;
        }
        MemShield shield2(buffer_res);
        
        IPCDataHeader header;
        header.msg_id = COMMAND_ID_FE_BE_DB_SERIES_LIST_RESULT;
        header.data_len = buffer_size;
        controller->get_client_proxy()->sync_send_data(header, buffer_res);
    }
    
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEDBGetSeriesList";
    return 0;
}

MED_IMG_END_NAMESPACE
