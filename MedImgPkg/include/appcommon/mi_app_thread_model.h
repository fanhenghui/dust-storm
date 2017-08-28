#ifndef MED_IMG_APP_THREAD_MODEL_H_
#define MED_IMG_APP_THREAD_MODEL_H_

#include "appcommon/mi_app_common_export.h"

#include <memory>
#include <deque>

#include "boost/thread/mutex.hpp"

MED_IMG_BEGIN_NAMESPACE 

class IPCClientProxy;
class IOperation;
class GLContext;
class AppController;
class AppCommon_Export AppThreadModel
{
public:
    AppThreadModel();
    ~AppThreadModel();

    void set_client_proxy(std::shared_ptr<IPCClientProxy> proxy);
    void set_controller(std::shared_ptr<AppController> controller);

    void push_operation(const std::shared_ptr<IOperation>& op);

    std::shared_ptr<GLContext> get_gl_context();
    
    void start();
    void stop();

protected:
    void process_operating();
    void process_rendering();
    void process_sending();

    void pop_operation(std::shared_ptr<IOperation>* op);

private:
    std::shared_ptr<IPCClientProxy> _proxy;

    std::shared_ptr<GLContext> _glcontext;

    std::weak_ptr<AppController> _controller;

    struct InnerThreadData;
    std::shared_ptr<InnerThreadData> _th_operating;
    std::shared_ptr<InnerThreadData> _th_rendering;
    std::shared_ptr<InnerThreadData> _th_sending;

    struct InnerQueue;
    std::unique_ptr<InnerQueue> _op_queue;

    std::deque<unsigned int> _dirty_cells;
    boost::mutex _dirty_cells_mutex;


    bool _rendering;
    bool _sending;

};

MED_IMG_END_NAMESPACE


#endif