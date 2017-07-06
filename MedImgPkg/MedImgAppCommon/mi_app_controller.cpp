#include "mi_app_controller.h"

#include "MedImgUtil/mi_ipc_client_proxy.h"

#include "mi_app_thread_model.h"

MED_IMG_BEGIN_NAMESPACE

AppController::AppController()
{
    _proxy.reset(new IPCClientProxy());
    _thread_model.reset(new AppThreadModel());
    _thread_model->set_client_proxy(_proxy);

    //process info
    _local_pid = static_cast<unsigned int>( getpid() );
    _server_pid = 0;
}

AppController::~AppController()
{

}

void AppController::initialize()
{

}

pid_t AppController::get_local_pid() const
{
    return _local_pid;
}

void AppController::set_server_pid(pid_t pid)
{
    _server_pid = pid;
}

pid_t AppController::get_server_pid() const
{
    return _server_pid;
}

void AppController::run(const std::string& path)
{
    _thread_model->start();
    _proxy->run();
}

std::shared_ptr<AppThreadModel> AppController::get_thread_model()
{
    return _thread_model;
}

std::shared_ptr<IPCClientProxy> AppController::get_client_proxy()
{
    return _proxy;
}

void AppController::add_cell(unsigned int id , std::shared_ptr<AppCell> cell)
{
    _cells[id] = cell;
}

void AppController::remove_cell(unsigned int id)
{
    //TODO release cell
    auto it = _cells.find(id);
    if(it != _cells.end()){
        _cells.erase(it);
    }
}

std::shared_ptr<AppCell> AppController::get_cell(unsigned int id)
{
    auto it = _cells.find(id);
    if(it != _cells.end()){
        return it->second;
    }
    else{
        return nullptr;
    }
}

std::map<unsigned int , std::shared_ptr<AppCell>> AppController::get_cells()
{
    return _cells;
}

MED_IMG_END_NAMESPACE