#ifndef WIN32

#include "mi_socket_client.h"

#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


MED_IMG_BEGIN_NAMESPACE


SocketClient::SocketClient():_accepted(false),_path("")
{

}

SocketClient::~SocketClient()
{

}

void SocketClient::register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler)
{
    _handler = handler;
}

std::string SocketClient::get_path() const
{
    return _path;
}

void SocketClient::set_path(const std::string& path)
{
    _path = path;
}

void SocketClient::run()
{

}


void SocketClient::send(const IPCDataHeader& data , void* buffer)
{

}

MED_IMG_END_NAMESPACE

#endif