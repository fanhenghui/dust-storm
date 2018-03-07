#include "mi_be_cmd_handler_fe_pacs_retrieve.h"

#include <boost/algorithm/string.hpp>

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "io/mi_protobuf.h"
#include "io/mi_configure.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_common_util.h"
#include "mi_app_common_logger.h"
#include "mi_model_pacs_cache.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPACSRetrieve::BECmdHandlerFEPACSRetrieve(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEPACSRetrieve::~BECmdHandlerFEPACSRetrieve() {}

int BECmdHandlerFEPACSRetrieve::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEPACSRetrieve";

    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelPACSCache> pacs_cache_model = AppCommonUtil::get_model_pacs_cache(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(pacs_cache_model);

    //check msg
    MsgDcmPACSRetrieveKey msg;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse series message from FE PACS fetch failed.";
        return -1;
    }    

    //Test parse to print
    if (msg.series_uid_size() != msg.study_uid_size()) {
        MI_APPCOMMON_LOG(MI_ERROR) << "series uid size not equal with study uid size.";
        return -1;
    }

    const int series_size = msg.series_uid_size();
    if (series_size <= 0) {
        MI_APPCOMMON_LOG(MI_WARNING) << "PACS retrieve null series.";
        return -1;
    }
    
    //--------------------------//
    //get real series/study uid
    //--------------------------//
    int study_idx = -1;
    int series_idx = -1;
    std::string study_uid,series_uid;
    for (int i = 0; i < series_size; ++i) {
        MI_APPCOMMON_LOG(MI_DEBUG) << "PACS retrieve series: " << msg.series_uid(i);
        study_idx = atoi(msg.study_uid(i).c_str());
        series_idx = atoi(msg.series_uid(i).c_str());
        
        if(0 != pacs_cache_model->get_study_series_uid(study_idx, series_idx, study_uid, series_uid)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "cant get series/study uid when retrieve.";
            return -1;
        }
        msg.set_study_uid(i, study_uid);
        msg.set_series_uid(i, series_uid);

        MI_APPCOMMON_LOG(MI_DEBUG) << "PACS retrieve study uid: " << study_uid << ". \n series uid: " << series_uid;

    }
    
    char* buffer_res = nullptr;
    int buffer_size = 0;
    if (0 != protobuf_serialize(msg, buffer_res, buffer_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialized study list msg buffer failed.";
        return -1;  
    }
    MemShield shield2(buffer_res);
    msg.Clear();

    //send message to DBS to fetch choosed DICOM series
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_PACS_RETRIEVE;
    header.data_len = buffer_size;
    IPCPackage* package = new IPCPackage(header,buffer_res);
    if(0 != controller->get_client_proxy_dbs()->sync_send_data(package)) {
        delete package;
        MI_APPCOMMON_LOG(MI_ERROR) << "send to DB to retrieve PACS failed.";
        return -1;
    }    

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEPACSRetrieve";
    return 0;
}

MED_IMG_END_NAMESPACE