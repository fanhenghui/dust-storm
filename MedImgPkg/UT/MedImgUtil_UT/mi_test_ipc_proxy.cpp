//#include <iostream>
//#include <string>
//
//#ifdef WIN32
//#include <Windows.h>
//#else
//#include <unistd.h>
//#endif
//
//#include "boost/thread.hpp"
//
//#include "util/mi_message_queue.h"
//
//
//int TestIPCProxy(int argc , char* argv[])
//{
//    _message_queue.activate();
//
//    boost::thread th(run);
//
//    boost::thread th2(mainloop);
//
//    th.join();
//    th2.join();
//
//    return 0;
//}