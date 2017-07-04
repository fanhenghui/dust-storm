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





MED_IMG_END_NAMESPACE