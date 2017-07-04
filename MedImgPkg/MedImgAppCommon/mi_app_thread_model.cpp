#include "mi_app_thread_model.h"

#include "boost/thread/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/condition.hpp"

#include "MedImgUtil/mi_message_queue.h"



MED_IMG_BEGIN_NAMESPACE

struct AppThreadModel::InnerThreadData
{
    boost::thread _th;
    boost::mutex _mutex;
    boost::condition _condition;
};

struct AppThreadModel::InnerQueue
{
    MessageQueue<std::shared_ptr<IOperation>> _msg_queue;
};

AppThreadModel::AppThreadModel()
{

}

AppThreadModel::~AppThreadModel()
{

}

void AppThreadModel::initialize()
{
    //
}

void AppThreadModel::finalize()
{

}

void AppThreadModel::set_client_proxy(std::shared_ptr<IPCClientProxy> proxy)
{
    _proxy = proxy;
}

void AppThreadModel::push_operation(const std::shared_ptr<IOperation>& op)
{
    _op_queue->_msg_queue.push(op);
}

void AppThreadModel::pop_operation(std::shared_ptr<IOperation>& op)
{
    _op_queue->_msg_queue.pop(op);
}

void AppThreadModel::start()
{

}

void AppThreadModel::process_operating()
{

}

void AppThreadModel::process_rendering()
{

}

void AppThreadModel::process_sending()
{

}




MED_IMG_END_NAMESPACE