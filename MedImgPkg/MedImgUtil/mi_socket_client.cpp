#ifndef WIN32

#include "mi_socket_client.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <iostream>

#include "mi_exception.h"


MED_IMG_BEGIN_NAMESPACE


SocketClient::SocketClient():_path(""),_fd_server(-1)
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
    if(path.size() > 108)
    {
        UTIL_THROW_EXCEPTION("Path length is too long!");
    }
    _path = path;
}

void SocketClient::run()
{
    int fd_s = socket(AF_UNIX , SOCK_STREAM , 0);
    if(fd_s == -1) {
        UTIL_THROW_EXCEPTION("create socket failed!");
    }

    struct sockaddr_un remote;
    remote.sun_family = AF_UNIX;
    for(size_t i = 0 ; i< _path.size() ; ++i){
        remote.sun_path[i] = _path[i];
    }
    socklen_t len = sizeof(remote);

    /////////////////////////////////////////////////////////
    //这里连了20次，每一秒连一次
    int connect_status = -1;
    for(int i = 0 ; i< _reconnect_times; ++i){
        connect_status = connect(fd_s , (struct sockaddr*)(&remote) ,len );
        if(connect_status != -1){
            break;
        }
        std::cout << "connecting times : " << i << std::endl;
        sleep(1);
    }
    /////////////////////////////////////////////////////////

    if(connect_status == -1) {
        UTIL_THROW_EXCEPTION("Connect server failed");
    }

    _fd_server = fd_s;

    
    for(;;) {

        IPCDataHeader header;
        void* buffer = nullptr;
        if(-1 == recv(_fd_server , &header , sizeof(header) , 0)) {
            std::cout << "warning recv failed!\n";
            //TODO ERROR Handle
            continue;
        }

        if(header._data_len <= 0) {
            //TODO WARNING Handle
            std::cout << "buffer length is less than 0";
        }
        else
        {
            buffer = new char[header._data_len];
            if(-1 == recv(_fd_server ,buffer , header._data_len , 0) ) {
                std::cout << "warning recv failed!\n";
                //TODO ERROR Handle
                continue;
            }
        }
        
        
        if(_handler && -1 == _handler->handle(header , buffer))
        {
            //TODO quit id
            break;
        }
    }
}

void SocketClient::send_data(const IPCDataHeader& dataheader , void* buffer)
{
    if(-1 == _fd_server)
    {
        //TODO ERROR Handle
        return;
    }

    if(-1 == send(_fd_server , &dataheader , sizeof(dataheader) , 0)) {
            return;
    }

    if(buffer != nullptr && dataheader._data_len > 0 ){
        if(-1 == send(_fd_server , buffer ,dataheader._data_len , 0)) {
            return;
        }
    }
}

MED_IMG_END_NAMESPACE

#endif