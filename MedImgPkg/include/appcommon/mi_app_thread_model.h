#ifndef MED_IMG_APPCOMMON_MI_APP_THREAD_MODEL_H
#define MED_IMG_APPCOMMON_MI_APP_THREAD_MODEL_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_message_queue.h"

#include <memory>
#include <deque>

#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/condition.hpp>

MED_IMG_BEGIN_NAMESPACE

class IPCClientProxy;
class IOperation;
class GLContext;
class AppController;
class AppCommon_Export AppThreadModel {
public:
    AppThreadModel();
    ~AppThreadModel();

    void set_controller(std::shared_ptr<AppController> controller);

    void set_client_proxy(std::shared_ptr<IPCClientProxy> proxy);
    void push_operation(const std::shared_ptr<IOperation>& op);

    void set_client_proxy_dbs(std::shared_ptr<IPCClientProxy> proxy);
    void push_operation_dbs(const std::shared_ptr<IOperation>& op);

    std::shared_ptr<GLContext> get_gl_context();

    void start(const std::string& unix_path);
    void stop();

protected:
    void process_operating();
    void process_rendering();
    void process_sending();
    void pop_operation(std::shared_ptr<IOperation>* op);

    void process_dbs_recving();
    void process_dbs_operation();

private:
    std::shared_ptr<IPCClientProxy> _proxy;

    std::shared_ptr<GLContext> _glcontext;

    std::weak_ptr<AppController> _controller;

    //for Node server
    struct InnerThreadData;
    std::shared_ptr<InnerThreadData> _th_operating;
    std::shared_ptr<InnerThreadData> _th_rendering;
    std::shared_ptr<InnerThreadData> _th_sending;

    struct InnerQueue;
    std::unique_ptr<InnerQueue> _op_queue;

    std::deque<unsigned int> _dirty_images;
    boost::mutex _dirty_images_mutex;

    std::deque<unsigned int> _dirty_none_images;
    boost::mutex _dirty_none_images_mutex;

    bool _rendering;
    bool _sending;

    //for DBS
    std::shared_ptr<IPCClientProxy> _client_proxy_dbs;
    boost::thread _thread_dbs_recving;
    boost::thread _thread_dbs_operating;
    MessageQueue<std::shared_ptr<IOperation>> _op_msg_queue_dbs;
};

MED_IMG_END_NAMESPACE


#endif