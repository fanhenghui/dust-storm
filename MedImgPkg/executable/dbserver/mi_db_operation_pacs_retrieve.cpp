#include "mi_db_operation_pacs_retrieve.h"

#include "io/mi_pacs_communicator.h"

#include "appcommon/mi_message.pb.h"

#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

DBOpPACSRetrieve::DBOpPACSRetrieve() {

}

DBOpPACSRetrieve::~DBOpPACSRetrieve() {

}

int DBOpPACSRetrieve::execute() {    
    std::shared_ptr<DBServerController> controller = get_controller<DBServerController>();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<DBEvaluationDispatcher> dispatcher = controller->get_evaluation_dispatcher();
    DBSERVER_CHECK_NULL_EXCEPTION(dispatcher);

    std::shared_ptr<PACSCommunicator> pacs_commu = controller->get_pacs_communicator();
    DBSERVER_CHECK_NULL_EXCEPTION(pacs_commu);

    std::vector<DcmInfo> dcm_infos;
    if (0 != pacs_commu->retrieve_all_series(dcm_infos)) {
        MI_DBSERVER_LOG(MI_ERROR) << "PACS retrive all series failed.";
        //TODO send message to notify BE
        return -1; 
    }

    //DEBUG
    int id = 0;
    for (auto it = dcm_infos.begin(); it != dcm_infos.end(); ++it) {
        const std::string series_id = (*it).series_id;
        MI_DBSERVER_LOG(MI_DEBUG) << id++ << " " << series_id;
    }
    
    return 0;
}

MED_IMG_END_NAMESPACE