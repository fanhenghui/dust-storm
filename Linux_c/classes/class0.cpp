#include "class0.h"
#include <boost/thread.hpp>
//#define TEST

namespace
{
    void run()
    {
#ifdef TEST
        std::cout << "This is test.\n";
#endif
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
