#include "mi_app_controller.h"

#include "util/mi_ipc_client_proxy.h"
#include "glresource/mi_gl_context.h"
#include "renderalgo/mi_volume_infos.h"

#include "mi_app_thread_model.h"
#include "mi_app_common_define.h"
#include "util/mi_model_interface.h"

MED_IMG_BEGIN_NAMESPACE

AppController::AppController() {
    _proxy.reset(new IPCClientProxy(UNIX));
    _client_proxy_dbs.reset(new IPCClientProxy(INET));
    _thread_model.reset(new AppThreadModel());
    _thread_model->set_client_proxy_fe(_proxy);
    _thread_model->set_client_proxy_dbs(_client_proxy_dbs);

    //process info
    _local_pid = static_cast<unsigned int>(getpid());
    _server_pid = 0;
}

AppController::~AppController() {

}

void AppController::initialize() {
    _thread_model->set_controller(shared_from_this());
}

void AppController::finalize() {
    _thread_model->stop();
}

pid_t AppController::get_local_pid() const {
    return _local_pid;
}

void AppController::set_server_pid(pid_t pid) {
    _server_pid = pid;
}

pid_t AppController::get_server_pid() const {
    return _server_pid;
}

void AppController::run(const std::string& unix_path) {
    _thread_model->run(unix_path);
}

std::shared_ptr<AppThreadModel> AppController::get_thread_model() {
    return _thread_model;
}

std::shared_ptr<IPCClientProxy> AppController::get_client_proxy() {
    return _proxy;
}

std::shared_ptr<IPCClientProxy> AppController::get_client_proxy_dbs() {
    return _client_proxy_dbs;
}

void AppController::add_cell(unsigned int id , std::shared_ptr<AppCell> cell) {
    _cells[id] = cell;
}

void AppController::remove_cell(unsigned int id) {
    auto it = _cells.find(id);
    if (it != _cells.end()) {
        _cells.erase(it);
    }
}

void AppController::remove_all_cells() {
    _cells.clear();
}

std::shared_ptr<AppCell> AppController::get_cell(unsigned int id) const {
    std::map<unsigned int, std::shared_ptr<AppCell>>::const_iterator it = _cells.find(id);
    if (it != _cells.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

std::map<unsigned int, std::shared_ptr<AppCell>> AppController::get_cells() const {
    return _cells;
}


void AppController::set_volume_infos(
    std::shared_ptr<VolumeInfos> volumeinfos) {
    _volumeinfos = volumeinfos;
}

std::shared_ptr<VolumeInfos> AppController::get_volume_infos() const {
    return _volumeinfos;
}

void AppController::add_model(unsigned int id, std::shared_ptr<IModel> model) {
    _models[id] = model;
}

void AppController::remove_model(unsigned int id) {
    auto it = _models.find(id);
    if (it != _models.end()) {
        _models.erase(it);
    }
}

void AppController::remove_all_models() {
    _models.clear();
}

std::shared_ptr<IModel> AppController::get_model(unsigned int id) const {
    std::map<unsigned int, std::shared_ptr<IModel>>::const_iterator it = _models.find(id);
    if (it != _models.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

std::map<unsigned int, std::shared_ptr<IModel>> AppController::get_modelss() const {
    return _models;
}

MED_IMG_END_NAMESPACE