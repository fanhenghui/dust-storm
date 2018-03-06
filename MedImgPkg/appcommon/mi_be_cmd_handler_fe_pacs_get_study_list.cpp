#include "mi_be_cmd_handler_fe_pacs_get_study_list.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "mi_app_common_define.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_model_pacs_cache.h"
#include "mi_app_common_util.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPACSGetStudyList::BECmdHandlerFEPACSGetStudyList(
    std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEPACSGetStudyList::~BECmdHandlerFEPACSGetStudyList() {}

int BECmdHandlerFEPACSGetStudyList::handle_command(const IPCDataHeader &dataheader, char *buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEPACSGetStudyList";

    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    MemShield shield(buffer);

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelPACSCache> pacs_cache_model = AppCommonUtil::get_model_pacs_cache(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(pacs_cache_model);

    MsgListPage msg_list_page;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg_list_page)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "encode study list page failed."; 
        return -1;
    }
    const int study_from = msg_list_page.from();
    const int study_to = msg_list_page.to();
    
    std::vector<MsgStudyInfo*> study_infos;
    std::vector<MsgPatientInfo*> patient_infos;
    pacs_cache_model->get_study_infos(study_from, study_to, study_infos, patient_infos);
    if (!study_infos.empty()) {
        MsgStudyWrapperCollection msg_res;
        for (size_t i = 0; i < study_infos.size(); ++i) {
            MsgStudyWrapper* study_wrapper = msg_res.add_study_wrappers();
            MsgStudyInfo* study_info = study_wrapper->mutable_study_info();
            *study_info = *(study_infos[i]);
            MsgPatientInfo* patient_info = study_wrapper->mutable_patient_info();
            *patient_info = *(patient_infos[i]);
        }

        char* buffer_res = nullptr;
        int buffer_size = 0;
        if (0 != protobuf_serialize(msg_res, buffer_res, buffer_size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "serialized study list msg buffer failed.";
            return -1;
        }
        MemShield shield2(buffer_res);
        
        IPCDataHeader header;
        header.msg_id = COMMAND_ID_FE_BE_PACS_STUDY_LIST_RESULT;
        header.data_len = buffer_size;
        controller->get_client_proxy()->sync_send_data(header, buffer_res);
    }
    
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEPACSGetStudyList";
    return 0;
}

MED_IMG_END_NAMESPACE
