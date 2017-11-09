#ifndef MEDIMG_MI_DB_SERVER_CONTROLLER_H
#define MEDIMG_MI_DB_SERVER_CONTROLLER_H

#include "mi_db_server_logger.h"
#include "mi_db_server_common.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class DBServerThreadModel;
class IPCServerProxy;
class DB;
class DBServerController : public std::enable_shared_from_this<DBServerController>{
public:
    DBServerController();
    ~DBServerController();

    void initialize();
    void run();
    void finalize();

    std::shared_ptr<DBServerThreadModel> get_thread_model();
    std::shared_ptr<IPCServerProxy> get_server_proxy();
    std::shared_ptr<DB> get_db();

private:
    void connect_db_i();

private:
    std::shared_ptr<DB> _db;
    std::shared_ptr<DBServerThreadModel> _thread_model;
    std::shared_ptr<IPCServerProxy> _server_proxy;
};

MED_IMG_END_NAMESPACE

#endif