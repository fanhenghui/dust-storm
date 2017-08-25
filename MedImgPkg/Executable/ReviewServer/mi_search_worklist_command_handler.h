#ifndef MED_IMG_SEARCH_WORKLIST_COMMAND_HANDLER_H
#define MED_IMG_SEARCH_WORKLIST_COMMAND_HANDLER_H

#include "MedImgAppCommon/mi_app_common_export.h"
#include "MedImgUtil/mi_ipc_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppController;

class SearchWorklistCommandHandler : public ICommandHandler {
public:
    SearchWorklistCommandHandler(std::shared_ptr<AppController> controller);
    virtual ~SearchWorklistCommandHandler();

    virtual int handle_command(const IPCDataHeader &datahaeder, char *buffer);

private:
    void ConstructWorklist();
    
private:
    std::weak_ptr<AppController> _controller;
};
MED_IMG_END_NAMESPACE

#endif