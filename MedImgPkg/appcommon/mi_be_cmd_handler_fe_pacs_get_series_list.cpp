#include "mi_be_cmd_handler_fe_pacs_get_series_list.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_protobuf.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_model_pacs_cache.h"
#include "mi_app_common_util.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPACSGetSeriesList::BECmdHandlerFEPACSGetSeriesList(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEPACSGetSeriesList::~BECmdHandlerFEPACSGetSeriesList() {}

int BECmdHandlerFEPACSGetSeriesList::handle_command(const IPCDataHeader &dataheader, char *buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEPACSGetSeriesList";

    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    MemShield shield(buffer);

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelPACSCache> pacs_cache_model = AppCommonUtil::get_model_pacs_cache(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(pacs_cache_model);

    MsgInt msg_study_idx;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg_study_idx)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "encode study idx failed."; 
        return -1;
    }
    const int study_idx = msg_study_idx.value();
    msg_study_idx.Clear();

    std::vector<MsgSeriesInfo*> series_infos;
    pacs_cache_model->get_series_info(study_idx, series_infos);
    if (!series_infos.empty()) {
        MsgStudyWrapper msg_res;
        for (size_t i = 0; i < series_infos.size(); ++i) {
            MsgSeriesInfo* series_info = msg_res.add_series_infos();
            *series_info = *(series_infos[i]);
            series_info->set_series_uid("");
        }

        char* buffer_res = nullptr;
        int buffer_size = 0;
        if (0 != protobuf_serialize(msg_res, buffer_res, buffer_size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "serialized series list msg buffer failed.";
            return -1;
        }
        MemShield shield2(buffer_res);
        
        IPCDataHeader header;
        header.msg_id = COMMAND_ID_FE_BE_PACS_SERIES_LIST_RESULT;
        header.data_len = buffer_size;
        controller->get_client_proxy()->sync_send_data(header, buffer_res);
    }
    
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEPACSGetSeriesList";
    return 0;
}

MED_IMG_END_NAMESPACE
