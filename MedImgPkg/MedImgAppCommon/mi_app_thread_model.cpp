#include "mi_app_thread_model.h"

#include "boost/thread/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/condition.hpp"

#include "MedImgUtil/mi_message_queue.h"
#include "MedImgGLResource/mi_gl_resource_manager_container.h"
#include "MedImgGLResource/mi_gl_context.h"

#include "mi_operation_interface.h"


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

AppThreadModel::AppThreadModel():
    _rendering(false),
    _sending(false),
    _th_rendering(new InnerThreadData()),
    _th_sending(new InnerThreadData()),
    _th_operating(new InnerThreadData()),
    _op_queue(new InnerQueue())
{
    //Creare gl context
    UIDType uid(0);
    _glcontext = GLResourceManagerContainer::instance()->get_context_manager()->create_object(uid);
    _glcontext->initialize();
}

AppThreadModel::~AppThreadModel()
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

void AppThreadModel::pop_operation(std::shared_ptr<IOperation>* op)
{
    _op_queue->_msg_queue.pop(op);
}

void AppThreadModel::start()
{
    try
    {
        _th_operating->_th = boost::thread(boost::bind(&AppThreadModel::process_operating, this));
            
        _th_sending->_th = boost::thread(boost::bind(&AppThreadModel::process_sending, this));

        _th_rendering->_th = boost::thread(boost::bind(&AppThreadModel::process_rendering, this));
    }
    catch(...)
    {
        //TODO ERROR
    }
    
}

void AppThreadModel::stop()
{
    _th_rendering->_th.interrupt();
    _th_rendering->_condition.notify_one();

    _th_sending->_th.interrupt();
    _th_sending->_condition.notify_one();

    _th_operating->_th.interrupt();
    _th_operating->_condition.notify_one();

    _th_rendering->_th.join();
    _th_sending->_th.join();
    _th_operating->_th.join();
}

void AppThreadModel::process_operating()
{
    try
    {
        for(;;){
            std::shared_ptr<IOperation> op;
            this->pop_operation(&op);

            boost::mutex::scoped_lock locker(_th_rendering->_mutex);

            int err = op->execute();
            if(-1 == err){
                //TODO execute failed

            }

            //interrupt point    
            boost::this_thread::interruption_point();

            _rendering = true;
            _th_rendering->_condition.notify_one();       
        }
        
    }
    catch(const Exception& e)
    {   
        //TODO ERROR
    }
    catch(boost::thread_interrupted& e)
    {
        //TODO thread interrupted 
    }
    catch(...)
    {
        
    }
}

void AppThreadModel::process_rendering()
{
    try
    {
        
        _glcontext->make_current();

        for(;;){

            ///\ 1 render
            {
                boost::mutex::scoped_lock locker(_th_rendering->_mutex);

                while(!_rendering){
                    _th_rendering->_condition.wait(_th_rendering->_mutex);
                }

                ////////////////////////////////////////
                //TODOTODO rendering code
                ////////////////////////////////////////

                //interrupt point    
                boost::this_thread::interruption_point();

                _rendering = false;

            }
            
            /// \2 get image result to buffer
            
            ////////////////////////////////////////
            //TODOTODO rendering code
            ////////////////////////////////////////
            _sending = true;    
            _th_sending->_condition.notify_one();
        }


        _glcontext->make_noncurrent();
    }
    catch(const Exception& e)
    {

    }
    catch(boost::thread_interrupted& e)
    {
        //TODO 
    }
    catch(...)
    {

    }
}

void AppThreadModel::process_sending()
{
    try
    {
        for(;;){

            ///\ sending image to fe by pic proxy
            boost::mutex::scoped_lock locker(_th_sending->_mutex);

            while(!_sending){
                _th_sending->_condition.wait(_th_sending->_mutex);
            }

            ////////////////////////////////////////
            //TODOTODO sending code
            ////////////////////////////////////////

            //interrupt point    
            boost::this_thread::interruption_point();

            _sending = false;
        }
    }
    catch(const Exception& e)
    {

    }
    catch(boost::thread_interrupted& e)
    {
        //TODO 
    }
    catch(...)
    {

    }
}




MED_IMG_END_NAMESPACE