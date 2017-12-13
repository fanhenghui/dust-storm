#ifndef MEDIMG_MI_DB_SERVER_CONTROLLER_H
#define MEDIMG_MI_DB_SERVER_CONTROLLER_H

#include "mi_db_server_logger.h"
#include "mi_db_server_common.h"
#include "util/mi_controller_interface.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class DBServerThreadModel;
class IPCServerProxy;
class DB;
class DBEvaluationDispatcher;
class PACSCommunicator;
class DBServerController : public IController, public std::enable_shared_from_this<DBServerController> {
public:
    DBServerController();
    virtual ~DBServerController();

    virtual void initialize();
    void run();
    void stop();
    virtual void finalize();

    std::shared_ptr<DBServerThreadModel> get_thread_model();
    std::shared_ptr<IPCServerProxy> get_server_proxy_be();
    std::shared_ptr<IPCServerProxy> get_server_proxy_ais();
    std::shared_ptr<DB> get_db();
    std::shared_ptr<DBEvaluationDispatcher> get_evaluation_dispatcher();
    std::shared_ptr<PACSCommunicator> get_pacs_communicator();

    void set_ais_client(unsigned int client);
    unsigned int get_ais_client() const;

private:
    void connect_db_i();

private:
    std::shared_ptr<DB> _db;
    std::shared_ptr<DBServerThreadModel> _thread_model;
    std::shared_ptr<IPCServerProxy> _server_proxy_be;
    std::shared_ptr<IPCServerProxy> _server_proxy_ais;
    std::shared_ptr<DBEvaluationDispatcher> _evaluation_dispatcher;
    //AIS socket ID
    unsigned int _ais_client;

    //PACS module
    std::shared_ptr<PACSCommunicator> _pacs_communicator;
};

MED_IMG_END_NAMESPACE

#endif