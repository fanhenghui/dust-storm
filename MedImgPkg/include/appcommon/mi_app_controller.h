#ifndef MED_IMG_APPCOMMON_MI_APP_CONTROLLER_H_
#define MED_IMG_APPCOMMON_MI_APP_CONTROLLER_H_

#include "appcommon/mi_app_common_export.h"
#include <map>
#include <memory>
#include <string>

MED_IMG_BEGIN_NAMESPACE

class AppThreadModel;
class IPCClientProxy;
class AppCell;
class VolumeInfos;
class IModel;
class AppCommon_Export AppController
        : public std::enable_shared_from_this<AppController> {
public:
    AppController();
    virtual ~AppController();

    virtual void initialize();
    void run(const std::string& unix_path);
    virtual void finalize();

    std::shared_ptr<AppThreadModel> get_thread_model();
    std::shared_ptr<IPCClientProxy> get_client_proxy();

    pid_t get_local_pid() const;
    void set_server_pid(pid_t pid);
    pid_t get_server_pid() const;

    // Cell container
    void add_cell(unsigned int id, std::shared_ptr<AppCell> cell);
    void remove_cell(unsigned int id);
    void remove_all_cells();
    std::shared_ptr<AppCell> get_cell(unsigned int id) const;
    std::map<unsigned int, std::shared_ptr<AppCell>> get_cells() const;

    //volume infos
    void set_volume_infos(std::shared_ptr<VolumeInfos> volumeinfos);
    std::shared_ptr<VolumeInfos> get_volume_infos() const;

    //models
    void add_model(unsigned int id, std::shared_ptr<IModel> model);
    void remove_model(unsigned int id);
    void remove_all_models();
    std::shared_ptr<IModel> get_model(unsigned int id) const;
    std::map<unsigned int, std::shared_ptr<IModel>> get_modelss() const;

protected:
    std::shared_ptr<IPCClientProxy> _proxy;
    std::shared_ptr<AppThreadModel> _thread_model;

    // Cells
    std::map<unsigned int, std::shared_ptr<AppCell>> _cells;
    // Models
    std::map<unsigned int, std::shared_ptr<IModel>> _models;

private:
    // process info
    pid_t _local_pid;
    pid_t _server_pid;

    std::shared_ptr<VolumeInfos> _volumeinfos;
};

MED_IMG_END_NAMESPACE

#endif