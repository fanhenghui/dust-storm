#include "class0.h"
#include <boost/thread.hpp>

namespace
{
    void run()
    {
        std::cout << "I am class 0 . in other thread.\n";
    }
}

Class0::Class0()
{

}

Class0::~Class0()
{

}

void Class0::test()
{
    boost::thread th(run);
    th.join();
}
