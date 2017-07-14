#include "mi_operation_factory.h"

MED_IMG_BEGIN_NAMESPACE

boost::mutex OperationFactory::_mutex;

OperationFactory* OperationFactory::_s_instance = nullptr;

OperationFactory* OperationFactory::instance()
{
    if (nullptr == _s_instance)
    {
        boost::unique_lock<boost::mutex> locker(_mutex);
        if (nullptr == _s_instance)
        {
            _s_instance = new OperationFactory();
        }
    }
    return _s_instance;
}

OperationFactory::~OperationFactory()
{

}

OperationFactory::OperationFactory()
{

}

std::shared_ptr<IOperation> OperationFactory::get_operation(unsigned int id)
{
    auto it = _ops.find(id);
    if(it != _ops.end()){
        return it->second;
    }
    else
    {
        return nullptr;
    }
}

void OperationFactory::register_operation(unsigned int id , std::shared_ptr<IOperation> op)
{
    _ops[id] = op;
}

MED_IMG_END_NAMESPACE