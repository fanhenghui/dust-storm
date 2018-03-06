#include "mi_be_cmd_handler_db_pacs_query_result.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"
#include "util/mi_operation_interface.h"

#include "io/mi_protobuf.h"
#include "io/mi_configure.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_util.h"
#include "mi_app_common_logger.h"
#include "mi_model_pacs_cache.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
class BEOpDBSendPACSQueryResult : public IOperation  {
public:
    BEOpDBSendPACSQueryResult() {};
    virtual ~BEOpDBSendPACSQueryResult() {};

    virtual int execute() {
        APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);
        std::shared_ptr<AppController> controller  = get_controller<AppController>();
        
        APPCOMMON_CHECK_NULL_EXCEPTION(controller);
        std::shared_ptr<IPCClientProxy> client_proxy = controller->get_client_proxy();

        IPCPackage* package = new IPCPackage(_header, _buffer);
        _buffer = nullptr;//move op buffer to IPC package

        if (0 != client_proxy->sync_send_data(package)) {
            delete package;
            package = nullptr;
            MI_APPCOMMON_LOG(MI_WARNING) << "send PACS query result failed.(server disconnected)";
        }

        return 0;
    }

    CREATE_EXTENDS_OP(BEOpDBSendPACSQueryResult)
};
}

BECmdHandlerDBPACSQueryResult::BECmdHandlerDBPACSQueryResult(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerDBPACSQueryResult::~BECmdHandlerDBPACSQueryResult() {}

int BECmdHandlerDBPACSQueryResult::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerDBPACSQueryResult";
    
    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    MemShield shield(buffer);

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //Debug code
    //Test parse and print
    MsgStudyWrapperCollection msg;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse study wrapper collection msg from DBS failed(PCAS query).";
        return -1;
    }

    std::shared_ptr<ModelPACSCache> pacs_cache_model = AppCommonUtil::get_model_pacs_cache(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(pacs_cache_model);
    pacs_cache_model->update(msg);
    
    int study_count = msg.num_study();
    msg.Clear();

    //for debug
    pacs_cache_model->print_all_series();

    std::vector<MsgStudyInfo*> study_infos;
    std::vector<MsgPatientInfo*> patient_infos;
    pacs_cache_model->get_study_infos(0, 10, study_infos, patient_infos);
    if (!study_infos.empty() && !patient_infos.empty()) {
        MsgStudyWrapperCollection msg_res;
        msg_res.set_num_study(study_count);
        MsgStudyWrapper* cur_study_wrapper = nullptr;
        for (size_t i = 0; i < study_infos.size(); ++i) {
            cur_study_wrapper = msg_res.add_study_wrappers();
            MsgStudyInfo* msg_study_info = cur_study_wrapper->mutable_study_info();
            *msg_study_info = *(study_infos[i]);
            MsgPatientInfo* msg_patient_info = cur_study_wrapper->mutable_patient_info();
            *msg_patient_info = *(patient_infos[i]);
        }

        char* buffer_res = nullptr;
        int buffer_size = 0;
        if (0 != protobuf_serialize(msg_res, buffer_res, buffer_size)) {
            MI_APPCOMMON_LOG(MI_ERROR) << "serialize dicom info collection message failed.";
            return -1;
        }
        msg_res.Clear();

        IPCDataHeader header;
        header.msg_id = COMMAND_ID_FE_BE_PACS_QUERY_RESULT;
        header.data_len = buffer_size;
        std::shared_ptr<IOperation> op(new BEOpDBSendPACSQueryResult());
        op->set_controller(controller);
        op->set_data(header , buffer_res);//transmit buffer from DB to FE(no need to shield)
        controller->get_thread_model()->push_operation_fe(op);
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerDBPACSQueryResult";
    return 0;
}

MED_IMG_END_NAMESPACE