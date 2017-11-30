#ifndef MEDIMGAIS_MI_AI_SERVER_CONTROLLER_H
#define MEDIMGAIS_MI_AI_SERVER_CONTROLLER_H

#include "mi_ai_server_logger.h"
#include "mi_ai_server_common.h"
#include "appcommon/mi_app_common_controller_interface.h"
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class AIServerThreadModel;
class IPCClientProxy;
class AIServerController : public IController, public std::enable_shared_from_this<AIServerController> {
public:
    AIServerController();
    virtual ~AIServerController();

    virtual void initialize();
    void run();
    virtual void finalize();

    std::shared_ptr<AIServerThreadModel> get_thread_model();
    std::shared_ptr<IPCClientProxy> get_client_proxy();

private:
    std::shared_ptr<AIServerThreadModel> _thread_model;
    std::shared_ptr<IPCClientProxy> _client_proxy;//for DBS
};

MED_IMG_END_NAMESPACE

#endif