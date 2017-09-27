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
#include "mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

SocketClient::SocketClient(): _path(""), _fd_server(-1) {

}

SocketClient::~SocketClient() {

}

void SocketClient::register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler) {
    _handler = handler;
}

std::string SocketClient::get_path() const {
    return _path;
}

void SocketClient::set_path(const std::string& path) {
    if (path.size() > 108) {
        MI_UTIL_LOG(MI_FATAL) << "SocketClient path length is too long(larger than 108).";
        UTIL_THROW_EXCEPTION("path length is too long(larger than 108).");
    }

    _path = path;
}

void SocketClient::run() {
    MI_UTIL_LOG(MI_TRACE) << "SocketClient("<< _path<< ")" << " running start.";

    const int fd_s = socket(AF_UNIX , SOCK_STREAM , 0);
    if (fd_s == -1) {
        MI_UTIL_LOG(MI_FATAL) << "SocketClient create socket failed.";
        UTIL_THROW_EXCEPTION("create socket failed.");
    }

    struct sockaddr_un remote;
    remote.sun_family = AF_UNIX;
    for (size_t i = 0; i < _path.size(); ++i) {
        remote.sun_path[i] = _path[i];
    }

    socklen_t len = sizeof(remote);

    ///\connect 100 times once per second
    int connect_status = -1;

    for (int i = 0; i < _reconnect_times; ++i) {
        connect_status = connect(fd_s , (struct sockaddr*)(&remote) , len);

        if (connect_status != -1) {
            break;
        }
        MI_UTIL_LOG(MI_INFO) << "trying connect to server times(once per second): " << i;
        sleep(1);
    }

    if (connect_status == -1) {
        MI_UTIL_LOG(MI_FATAL) << "connect server failed.";
        UTIL_THROW_EXCEPTION("connect server failed.");
    }

    _fd_server = fd_s;

    while (true) {

        IPCDataHeader header;
        char* buffer = nullptr;

        if (-1 == recv(_fd_server, &header, sizeof(header) , 0)) {
            MI_UTIL_LOG(MI_WARNING) << "client receive data header failed.";
            continue;
        }
        
        MI_UTIL_LOG(MI_TRACE) << "receive data header, " << STREAM_IPCHEADER_INFO(header);   

        if (header._data_len <= 0) {
            MI_UTIL_LOG(MI_TRACE) << "client received data buffer length less than 0.";
        } else {
            buffer = new char[header._data_len];
            if (-1 == recv(_fd_server, buffer, header._data_len, 0)) {
                MI_UTIL_LOG(MI_WARNING) << "client receive data buffer failed.";
                continue;
            }
        }

        try {
            if (_handler) {
                const int err = _handler->handle(header, buffer);
                if (err == CLIENT_QUIT_ID) {
                    break;
                }
            } else {
                MI_UTIL_LOG(MI_WARNING) << "client handler to process received data is null.";
            }

        } catch (const Exception& e) {
            //Ignore error to keep connecting
            MI_UTIL_LOG(MI_FATAL) << "handle command error(skip and continue): " << e.what();
        }
    }
    MI_UTIL_LOG(MI_TRACE) << "SocketClient("<< _path<< ")" << " running end.";
}

void SocketClient::send_data(const IPCDataHeader& dataheader , char* buffer) {

    MI_UTIL_LOG(MI_TRACE) << "SocketClient("<< _path<< ")" << " sending data: " << STREAM_IPCHEADER_INFO(dataheader);
    if (-1 == _fd_server) {
        MI_UTIL_LOG(MI_FATAL) << "send data: server fd invalid.";
        UTIL_THROW_EXCEPTION("send data: server fd invalid!");
        return;
    }

    //send header
    if (-1 == send(_fd_server , &dataheader , sizeof(dataheader) , 0)) {
        MI_UTIL_LOG(MI_WARNING) << "send data: failed to send data header. header detail: " << STREAM_IPCHEADER_INFO(dataheader);
        if (!buffer) {
            delete [] buffer;
            buffer = nullptr;
        }
        return;
    }

    //send context
    if (buffer != nullptr && dataheader._data_len > 0) {
        if (-1 == send(_fd_server , buffer , dataheader._data_len , 0)) {
            MI_UTIL_LOG(MI_WARNING) << "send data: failed to send data context. header detail: " << STREAM_IPCHEADER_INFO(dataheader);
            if (!buffer) {
                delete [] buffer;
                buffer = nullptr;
            }
            return;
        }
    }

    if (!buffer) {
        delete [] buffer;
        buffer = nullptr;
    }
}

MED_IMG_END_NAMESPACE

#endif