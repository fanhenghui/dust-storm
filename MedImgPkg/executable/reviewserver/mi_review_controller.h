#ifndef MED_IMG_REVIEW_CONTROLLER_H
#define MED_IMG_REVIEW_CONTROLLER_H

#include "util/mi_ipc_client_proxy.h"

#include <memory>
#include <string>

#include "appcommon/mi_app_controller.h"

MED_IMG_BEGIN_NAMESPACE

class ReviewController : public AppController {
public:
    ReviewController();
    virtual ~ReviewController();

    virtual void initialize();

private:
    void register_command_handler_i();
    void create_model_i();
private:
};

MED_IMG_END_NAMESPACE

#endif