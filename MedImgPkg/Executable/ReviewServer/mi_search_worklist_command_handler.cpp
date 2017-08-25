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
    MsgWorklist * worklist = this->createWorklist(); 
    // MsgWorklistItem * worklist = this->createWorklistItem();
    // serialize worklist, attach to header, and send
    int size = worklist->ByteSize();
    void * data = malloc(size);
    bool res = worklist->SerializeToArray(data, size);
    
    IPCDataHeader header;
    header._sender = static_cast<unsigned int>(controller->get_local_pid());
    header._receiver = static_cast<unsigned int>(controller->get_server_pid());
    header._msg_id = COMMAND_ID_BE_SEND_WORKLIST;
    header._data_type = 1; // protocol buffer

    if(!res)
    {
        size = 0;
        free(data);
        data = nullptr;
    }
    
    header._data_len = size;
    controller->get_client_proxy()->async_send_message(header, reinterpret_cast<char *>(data) );
    
    return 0;
}

// TODO: connect to pacs and do a real search
MsgWorklist * SearchWorklistCommandHandler::createWorklist()
{
    MsgWorklist* list = new MsgWorklist;
    for (int i=0; i<2; i++)
    {
        MsgWorklistItem *item = list->add_items();
        item->set_patient_id("pid_" + std::to_string(i+1));
        item->set_patient_name("pname_" + std::to_string(i+1));
        item->set_series_uid("sid_" + std::to_string(i+1));
        item->set_imaging_modality("CT");
    }
    return list;
}

MsgWorklistItem * SearchWorklistCommandHandler::createWorklistItem()
{
    MsgWorklistItem *item = new MsgWorklistItem;
    item->set_patient_id("pid");
    item->set_patient_name("pname");
    item->set_series_uid("sid");
    item->set_imaging_modality("CT");
    return item;
}

MED_IMG_END_NAMESPACE