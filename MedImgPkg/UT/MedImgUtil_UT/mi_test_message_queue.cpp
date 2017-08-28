#include <iostream>
#include <string>

#ifdef WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "boost/thread.hpp"

#include "util/mi_message_queue.h"

using namespace medical_imaging;


class Message
{
public:
    Message()
    {
    }
    ~Message()
    {

    }
    void set(const std::string& msg)
    {
        _msg = msg;
    }
    std::string get() const
    {
        return _msg;
    }
protected:
private:
    std::string _msg;
};

MessageQueue<Message> _message_queue;

void run()
{
    std::string s;
    while (std::cin >> s)
    {
        //Sleep(100);
        //std::cout << s << std::endl;
        Message msg;
        msg.set(s);
        _message_queue.push(msg);
    }
}

void mainloop()
{
    while(true)
    {
        Message msg;
        _message_queue.pop(&msg);

        std::cout << "main loop : " << msg.get() << std::endl;
    }
}

int TestMessageQueue(int argc , char* argv[])
{
    _message_queue.activate();

    boost::thread th(run);

    boost::thread th2(mainloop);

    th.join();
    th2.join();

    return 0;
}