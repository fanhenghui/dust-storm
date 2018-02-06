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

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerFEPACSRetrieve::BECmdHandlerFEPACSRetrieve(std::shared_ptr<AppController> controller)
    : _controller(controller) {}

BECmdHandlerFEPACSRetrieve::~BECmdHandlerFEPACSRetrieve() {}

int BECmdHandlerFEPACSRetrieve::handle_command(const IPCDataHeader& dataheader, char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerFEPACSRetrieve";

    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //check msg
    MsgDcmPACSRetrieveKey msg;
    if (0 != protobuf_parse(buffer, dataheader.data_len, msg)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse series message from FE PACS fetch failed.";
        return -1;
    }

    //Test parse to print
    const int series_size = msg.series_uid_size();
    for (int i = 0; i < series_size; ++i) {
        MI_APPCOMMON_LOG(MI_DEBUG) << "PACS retrieve series: " << msg.series_uid(i);
    }
    msg.Clear();

    if (series_size <= 0) {
        MI_APPCOMMON_LOG(MI_WARNING) << "PACS retrieve null series.";
        return -1;
    }

    //send message to DBS to fetch choosed DICOM series
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_PACS_RETRIEVE;
    header.data_len = dataheader.data_len;
    IPCPackage* package = new IPCPackage(header,buffer);
    if(0 != controller->get_client_proxy_dbs()->sync_send_data(package)) {
        delete package;
        MI_APPCOMMON_LOG(MI_ERROR) << "send to DB to retrieve PACS failed.";
        return -1;
    }    

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerFEPACSRetrieve";
    return 0;
}

MED_IMG_END_NAMESPACE