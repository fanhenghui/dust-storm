#include "mi_app_controller.h"

#include "MedImgUtil/mi_ipc_client_proxy.h"

#include "mi_app_thread_model.h"

MED_IMG_BEGIN_NAMESPACE

AppController::AppController()
{
    _proxy.reset(new IPCClientProxy());
    _thread_model.reset(new AppThreadModel());
    _thread_model->set_client_proxy(_proxy);
}

AppController::~AppController()
{

}

void AppController::initialize()
{

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

MED_IMG_END_NAMESPACE