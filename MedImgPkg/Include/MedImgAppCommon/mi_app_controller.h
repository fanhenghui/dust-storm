#ifndef MED_IMG_APP_CONTROLLER_H_
#define MED_IMG_APP_CONTROLLER_H_

#include "MedImgAppCommon/mi_app_common_export.h"
#include <string>
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AppThreadModel;
class IPCClientProxy;
class AppController : public std::enable_shared_from_this<AppController>
{
public:
    AppController();
    virtual ~AppController();

    virtual void initialize();
    void run(const std::string& path);

    std::shared_ptr<AppThreadModel> get_thread_model();
    std::shared_ptr<IPCClientProxy> get_client_proxy();

protected:
private:    
    std::shared_ptr<IPCClientProxy> _proxy;
    std::shared_ptr<AppThreadModel> _thread_model;
};

MED_IMG_END_NAMESPACE


#endif