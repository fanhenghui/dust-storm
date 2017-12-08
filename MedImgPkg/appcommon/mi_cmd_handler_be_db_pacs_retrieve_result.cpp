#include "mi_cmd_handler_be_db_pacs_retrieve_result.h"

#include "util/mi_ipc_client_proxy.h"
#include "util/mi_memory_shield.h"

#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_thread_model.h"

#include "mi_operation_interface.h"
#include "mi_message.pb.h"
#include "mi_app_config.h"
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

CmdHandlerBE_DBPACSRetrieveResult::CmdHandlerBE_DBPACSRetrieveResult(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

CmdHandlerBE_DBPACSRetrieveResult::~CmdHandlerBE_DBPACSRetrieveResult() {}

int CmdHandlerBE_DBPACSRetrieveResult::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN CmdHandler PACS retrieve response";
    APPCOMMON_CHECK_NULL_EXCEPTION(buffer); 
    MemShield shield(buffer);
    
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //receive DB's retrive response , and create operation to BE queue to nofity FE to refresh PACS table
    //Test
    MsgDcmInfoCollection msg;
    if (!msg.ParseFromArray(buffer, dataheader.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse dicom info collection msg from DBS failed(PCAS retrieve).";
    }
    msg.Clear();

    //transmit to FE 
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_FE_BE_PACS_RETRIEVE_RESULT;
    header.data_len = dataheader.data_len;
    std::shared_ptr<IOperation> op(new BEOpDBSendPACSRetrieveResult());
    op->set_controller(controller);
    op->set_data(header , buffer);
    controller->get_thread_model()->push_operation(op);

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT CmdHandler PACS retrieve response";
    return 0;
}

MED_IMG_END_NAMESPACE