#ifndef MED_IMG_APP_THREAD_MODEL_H_
#define MED_IMG_APP_THREAD_MODEL_H_

#include "MedImgAppCommon/mi_app_common_export.h"

#include <memory>

MED_IMG_BEGIN_NAMESPACE

class IPCClientProxy;
class IOperation;

class AppThreadModel
{
public:
    AppThreadModel();
    ~AppThreadModel();

    void start();

    void set_client_proxy(std::shared_ptr<IPCClientProxy> proxy);

    void push_operation(const std::shared_ptr<IOperation>& op);
    void pop_operation(std::shared_ptr<IOperation>& op);


protected:
    void process_operating();
    void process_rendering();
    void process_sending();

private:
    struct InnerThreadData;
    std::shared_ptr<InnerThreadData> _th_operate;
    std::shared_ptr<InnerThreadData> _th_render;
    std::shared_ptr<InnerThreadData> _th_send;

    struct InnerQueue;
    std::unique_ptr<InnerQueue> _op_queue;
    
    std::shared_ptr<IPCClientProxy> _proxy;
};

MED_IMG_END_NAMESPACE


#endif