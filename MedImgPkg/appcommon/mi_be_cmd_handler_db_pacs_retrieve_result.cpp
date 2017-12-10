#include "mi_be_cmd_handler_db_pacs_retrieve_result.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"
#include "util/mi_operation_interface.h"

#include "io/mi_message.pb.h"
#include "io/mi_configure.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_util.h"
#include "mi_model_pacs.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
class BEOpDBSendPACSRetrieveResult : public IOperation  {
public:
    BEOpDBSendPACSRetrieveResult() {};
    virtual ~BEOpDBSendPACSRetrieveResult() {};

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
            MI_APPCOMMON_LOG(MI_WARNING) << "send PACS retrieve result failed.(server disconnected)";
        }

        return 0;
    }

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<BEOpDBSendPACSRetrieveResult>(new BEOpDBSendPACSRetrieveResult());
    }

};
}

BECmdHandlerDBPACSRetrieveResult::BECmdHandlerDBPACSRetrieveResult(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerDBPACSRetrieveResult::~BECmdHandlerDBPACSRetrieveResult() {}

int BECmdHandlerDBPACSRetrieveResult::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS retrieve response";
    APPCOMMON_CHECK_NULL_EXCEPTION(buffer); 
    
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //parse PACS result and save to model
    MsgDcmInfoCollection msg;
    if (!msg.ParseFromArray(buffer, dataheader.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse dicom info collection msg from DBS failed(PCAS retrieve).";
    }
    std::vector<DcmInfo> dcm_infos(msg.dcminfo_size());
    int id=0;
    for (int i=0; i<msg.dcminfo_size(); ++i) {
        MsgDcmInfo item = msg.dcminfo(i);
        MI_APPCOMMON_LOG(MI_DEBUG) << "<><><><><><>series specification:<><><><><><>";
        MI_APPCOMMON_LOG(MI_DEBUG) << "study id: " << item.study_id();
        MI_APPCOMMON_LOG(MI_DEBUG) << "series id: " << item.series_id();
        MI_APPCOMMON_LOG(MI_DEBUG) << "study date: " << item.study_date(); 
        MI_APPCOMMON_LOG(MI_DEBUG) << "study time: " << item.study_time(); 
        MI_APPCOMMON_LOG(MI_DEBUG) << "patient id: " << item.patient_id(); 
        MI_APPCOMMON_LOG(MI_DEBUG) << "patient name: " << item.patient_name(); 
        MI_APPCOMMON_LOG(MI_DEBUG) << "patient sex: " << item.patient_sex();
        MI_APPCOMMON_LOG(MI_DEBUG) << "patient age: " << item.patient_age(); 
        MI_APPCOMMON_LOG(MI_DEBUG) << "patient birth date: " << item.patient_birth_date(); 
        MI_APPCOMMON_LOG(MI_DEBUG) << "modality: " << item.modality(); 
        
        DcmInfo dcm_info;
        dcm_info.study_id = item.study_id();
        dcm_info.series_id = item.series_id();
        dcm_info.study_date = item.study_date();
        dcm_info.study_time = item.study_time();
        dcm_info.patient_id = item.patient_id();
        dcm_info.patient_name = item.patient_name();
        dcm_info.patient_sex = item.patient_sex();
        dcm_info.patient_age = item.patient_age();
        dcm_info.patient_birth_date = item.patient_birth_date();
        dcm_infos[id++] = dcm_info;
    }
    msg.Clear();

    std::shared_ptr<ModelPACS> model_pacs = AppCommonUtil::get_model_pacs(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_pacs);
    model_pacs->update(dcm_infos);

    //receive DB's retrive response , and create operation to BE queue to nofity FE to refresh PACS table
    //transmit to FE 
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_FE_BE_PACS_RETRIEVE_RESULT;
    header.data_len = dataheader.data_len;
    std::shared_ptr<IOperation> op(new BEOpDBSendPACSRetrieveResult());
    op->set_controller(controller);
    op->set_data(header , buffer);//transmit buffer from DB to FE(no need to shield)
    controller->get_thread_model()->push_operation(op);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS retrieve response";
    return 0;
}

MED_IMG_END_NAMESPACE