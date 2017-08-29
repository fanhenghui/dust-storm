#ifndef WIN32
#ifndef MED_IMG_SOCKET_SERVER_H_
#define MED_IMG_SOCKET_SERVER_H_

#include "util/mi_util_export.h"
#include "util/mi_message_queue.h"

MED_IMG_BEGIN_NAMESPACE


class SocketServer {
public:
    SocketServer() {}
    ~SocketServer() {}

    // void push_command();
    // void quit();
    // void run();


protected:
private:
    std::shared_ptr<MessageQueue> _message_queue;
};

MED_IMG_END_NAMESPACE

#endif
#endif