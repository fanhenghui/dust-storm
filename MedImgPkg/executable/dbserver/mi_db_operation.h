#ifndef MED_IMG_MI_DB_OPERATION_H
#define MED_IMG_MI_DB_OPERATION_H

#include "appcommon/mi_operation_interface.h"
#include "mi_db_server_common.h"

MED_IMG_BEGIN_NAMESPACE
class DBServerController;
class DBOperation : public IOperation {
public:
    DBOperation() {}
    virtual ~DBOperation() {}

    void set_db_server_controller(std::shared_ptr<DBServerController> controller) {
        _db_server_controller = controller;
    } 

protected:
    std::weak_ptr<DBServerController> _db_server_controller;
};

MED_IMG_END_NAMESPACE

#endif