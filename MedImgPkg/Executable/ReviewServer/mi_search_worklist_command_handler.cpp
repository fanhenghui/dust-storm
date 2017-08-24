#include "mi_search_worklist_command_handler.h"

#include "mi_message.pb.h"
#include "MedImgAppCommon/mi_app_controller.h"
#include "MedImgAppCommon/mi_app_common_define.h"
#include "MedImgUtil/mi_ipc_client_proxy.h"

#include <iostream>

MED_IMG_BEGIN_NAMESPACE

SearchWorklistCommandHandler::SearchWorklistCommandHandler(std::shared_ptr<AppController> controller)
    : _controller(controller) {}
    
SearchWorklistCommandHandler::~SearchWorklistCommandHandler() {}

int SearchWorklistCommandHandler::handle_command(const IPCDataHeader &datahaeder, char *buffer)
{
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
      APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }
  
    std::cout << "Received Msg Searck Worklist" << std::endl;
    this->ConstructWorklist(); // should in the format of protocol buffer

    IPCDataHeader header;
    header._msg_id = COMMAND_ID_BE_SEND_WORKLIST;
    controller->get_client_proxy()->async_send_message(header, nullptr);

    return 0;
}

void SearchWorklistCommandHandler::ConstructWorklist()
{
    MsgWorklist list;
    for (int i=0; i<1; i++)
    {
        MsgWorklistItem *item = list.add_items();
        item->set_patient_id("1");
        item->set_patient_name("name");
        item->set_series_uid("1.1");
        item->set_imaging_modality("CT");
    }
}
MED_IMG_END_NAMESPACE