#include "mi_concurrency.h"
#include "boost/thread.hpp"

MED_IMAGING_BEGIN_NAMESPACE

Concurrency* Concurrency::_s_instance = nullptr;

boost::mutex Concurrency::_s_mutex;

Concurrency* Concurrency::instance()
{
    if (!_s_instance)
    {
        boost::unique_lock<boost::mutex> locker(_s_mutex);
        if (!_s_instance)
        {
            _s_instance = new Concurrency();
        }
    }
    return _s_instance;
}

Concurrency::~Concurrency()
{

}

Concurrency::Concurrency()
{
    _hardware_concurrency = boost::thread::hardware_concurrency();
    _app_concurrency = _hardware_concurrency;
}

void Concurrency::set_app_concurrency(unsigned int value)
{
    _app_concurrency = value > _hardware_concurrency ? _hardware_concurrency : value;
}

unsigned int Concurrency::get_app_concurrency()
{
    return _app_concurrency;
}

unsigned int Concurrency::get_hardware_concurrency()
{
    return _hardware_concurrency;
}

MED_IMAGING_END_NAMESPACE