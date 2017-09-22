#ifndef MED_IMG_APP_CONTROLLER_H_
#define MED_IMG_APP_CONTROLLER_H_

#include "appcommon/mi_app_common_export.h"
#include <map>
#include <memory>
#include <string>

MED_IMG_BEGIN_NAMESPACE

class AppThreadModel;
class IPCClientProxy;
class AppCell;
class AppCommon_Export AppController
        : public std::enable_shared_from_this<AppController> {
public:
    AppController();
    virtual ~AppController();

    virtual void initialize();
    void run(const std::string& path);
    virtual void finalize();

    std::shared_ptr<AppThreadModel> get_thread_model();
    std::shared_ptr<IPCClientProxy> get_client_proxy();

    pid_t get_local_pid() const;
    void set_server_pid(pid_t pid);
    pid_t get_server_pid() const;

    // Cell container
    void add_cell(unsigned int id, std::shared_ptr<AppCell> cell);
    void remove_cell(unsigned int id);
    std::shared_ptr<AppCell> get_cell(unsigned int id);
    std::map<unsigned int, std::shared_ptr<AppCell>> get_cells();

protected:
    std::shared_ptr<IPCClientProxy> _proxy;
    std::shared_ptr<AppThreadModel> _thread_model;

    // Cells
    std::map<unsigned int, std::shared_ptr<AppCell>> _cells;

private:
    // process info
    pid_t _local_pid;
    pid_t _server_pid;
};

MED_IMG_END_NAMESPACE

#endif