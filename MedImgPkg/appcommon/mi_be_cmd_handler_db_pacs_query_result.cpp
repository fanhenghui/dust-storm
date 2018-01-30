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
#include "mi_model_pacs.h"
#include "mi_app_common_logger.h"

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
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //parse PACS result and save to model
    MsgDcmInfoCollection msg;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse dicom info collection msg from DBS failed(PCAS query).";
        return -1;
    }

    std::vector<DcmInfo> dcm_infos(msg.dcminfo_size());
    int id=0;
    for (int i=0; i<msg.dcminfo_size(); ++i) {
        MsgDcmInfo item = msg.dcminfo(i);
        
        DcmInfo dcm_info;
        dcm_info.study_id = item.study_id();
        dcm_info.series_id = item.series_id();
        dcm_info.study_date = item.study_date();
        dcm_info.study_time = item.study_time();
        dcm_info.patient_id = item.patient_id();
        dcm_info.patient_name = item.patient_name();
        dcm_info.patient_sex = item.patient_sex();
        dcm_info.patient_birth_date = item.patient_birth_date();
        dcm_info.modality = item.modality();
        dcm_info.instance_number = item.instance_number();
        dcm_info.accession_number = item.accession_number();

        dcm_infos[id++] = dcm_info;

        MI_APPCOMMON_LOG(MI_DEBUG) << id++ << 
            "study_id: " << dcm_info.study_id << std::endl <<
            "series_id: " << dcm_info.series_id << std::endl <<
            "study_date: " << dcm_info.study_date << std::endl <<
            "study_time: " << dcm_info.study_time << std::endl <<
            "study_date: " << dcm_info.study_date << std::endl <<
            "patient_id: " << dcm_info.patient_id << std::endl <<
            "patient_name: " << dcm_info.patient_name << std::endl <<
            "patient_sex: " << dcm_info.patient_sex << std::endl <<
            "patient_birth_date: " << dcm_info.patient_birth_date << std::endl <<
            "modality: " << dcm_info.modality << std::endl <<
            "instance_number: " << dcm_info.instance_number << std::endl <<
            "accession_number: " << dcm_info.accession_number << std::endl;
    }
    msg.Clear();

    std::shared_ptr<ModelPACS> model_pacs = AppCommonUtil::get_model_pacs(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_pacs);
    model_pacs->update(dcm_infos);

    //receive DB's retrive response , and create operation to BE queue to nofity FE to refresh PACS table
    //transmit to FE 
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_FE_BE_PACS_QUERY_RESULT;
    header.data_len = dataheader.data_len;
    std::shared_ptr<IOperation> op(new BEOpDBSendPACSQueryResult());
    op->set_controller(controller);
    op->set_data(header , buffer);//transmit buffer from DB to FE(no need to shield)
    controller->get_thread_model()->push_operation_fe(op);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerDBPACSQueryResult";
    return 0;
}

MED_IMG_END_NAMESPACE