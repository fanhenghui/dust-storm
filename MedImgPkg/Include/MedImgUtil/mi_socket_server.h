#ifndef MED_IMG_SOCKET_SERVER_H_
#define MED_IMG_SOCKET_SERVER_H_

#include "MedImgUtil/mi_util_export.h"
#include "MedImgUtil/mi_message_queue.h"

MED_IMG_BEGIN_NAMESPACE


class SocketServer
{
public:
    SocketServer();
    ~SocketServer();

    void push_command();
    void quit();
    void run();
    void handle


protected:
private:
    std::shared_ptr<MessageQueue> _message_queue;
};

MED_IMG_END_NAMESPACE

#endif