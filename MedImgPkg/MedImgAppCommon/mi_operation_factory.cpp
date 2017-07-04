#include "mi_operation_factory.h"

MED_IMG_BEGIN_NAMESPACE


////////////////////////////////////////
//Test
class TestOperation : public IOperation
{
public:
    TestOperation() {};
    virtual ~TestOperation() {};
    virtual int operate()
    {
        std::cout << "Hello test operation.";
        return 1;
    };
};

////////////////////////////////////////

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

std::shared_ptr<IOperation> OperationFactory::get_operation(unsigned int id)
{
    return std::shared_ptr<IOperation>(new TestOperation());
}

MED_IMG_END_NAMESPACE