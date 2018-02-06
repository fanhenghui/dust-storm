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

    //Debug code
    //Test parse and print
    MsgStudyWrapperCollection msg;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse study wrapper collection msg from DBS failed(PCAS query).";
        return -1;
    }

    const int study_size = msg.study_wrappers_size();
    for (int i = 0; i < study_size; ++i) {
        const MsgStudyWrapper& study_wrapper = msg.study_wrappers(i);
        const MsgStudyInfo& study_info = study_wrapper.study_info();
        const MsgPatientInfo& patient_info = study_wrapper.patient_info();

        int series_size = study_wrapper.series_infos_size();
        for (int j = 0; j < series_size; ++j) {
            const MsgSeriesInfo& series_info = study_wrapper.series_infos(j);
            MI_APPCOMMON_LOG(MI_DEBUG) << i <<": " << std::endl
            << "study_uid: " << study_info.study_uid() << std::endl
            << "study_id: " << study_info.study_id() << std::endl
            << "study_date: " << study_info.study_date() << std::endl
            << "study_time: " << study_info.study_time() << std::endl
            << "accession_no: " << study_info.accession_no() << std::endl
            << "study_desc: " << study_info.study_desc() << std::endl
            << "num_instance(study): " << study_info.num_instance() << std::endl
            << "num_series: " << study_info.num_series() << std::endl
            << "series_uid: " << series_info.series_uid() << std::endl
            << "series_no: " << series_info.series_no() << std::endl
            << "modality: " << series_info.modality() << std::endl
            << "series_desc: " << series_info.series_desc() << std::endl
            << "institution: " << series_info.institution() << std::endl
            << "num_instance(series): " << series_info.num_instance() << std::endl
            << "patient_id: " << patient_info.patient_id() << std::endl
            << "patient_name: " << patient_info.patient_name() << std::endl
            << "patient_sex: " << patient_info.patient_sex() << std::endl
            << "patient_birth_date: " << patient_info.patient_birth_date() << std::endl
            << std::endl;
        }
    }

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