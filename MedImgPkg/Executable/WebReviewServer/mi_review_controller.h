#ifndef MED_IMG_REVIEW_CONTROLLER_H_
#define MED_IMG_REVIEW_CONTROLLER_H_

#include "MedImgUtil/mi_ipc_client_proxy.h"

#include <string>
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class ReviewController
{
public:
    ReviewController();
    ~ReviewController();
    
    void initialize();
    void run(const std::string& path);

protected:
private:
    std::shared_ptr<IPCClientProxy> _proxy;
};

MED_IMG_END_NAMESPACE

#endif