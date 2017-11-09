#ifndef WIN32

#include "mi_socket_client.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netdb.h> 

#include <iostream>

#include "mi_exception.h"
#include "mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

SocketClient::SocketClient(SocketType type /*= UNIX*/): 
    _path(""), _fd_server(0), _socket_type(type), _alive(true) {

}

SocketClient::~SocketClient() {
    if(_fd_server != 0) {
        close(_fd_server);
        _fd_server = 0;
    }
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

void SocketClient::set_server_address(const std::string& ip, const std::string& port) {
    _server_ip = ip;
    _server_port = port;
}

void SocketClient::get_server_address(std::string& ip, std::string& port) const {
    ip = _server_ip;
    port = _server_port;
}

void SocketClient::run() {
    MI_UTIL_LOG(MI_TRACE) << "SocketClient("<< _path<< ")" << " running start.";

    //create scoket
    int fd_s = 0;
    if (UNIX == _socket_type) {
        fd_s = socket(AF_UNIX , SOCK_STREAM , 0);
        if (fd_s == 0) {
            MI_UTIL_LOG(MI_FATAL) << "SocketClient create UNIX socket failed.";
            UTIL_THROW_EXCEPTION("create UNIX socket failed.");
        }
    
        struct sockaddr_un remote;
        bzero((char*)(&remote), sizeof(remote));
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
        MI_UTIL_LOG(MI_INFO) << "UNIX socket connect success.\n";
    } else if (INET == _socket_type) {
        if (_server_ip.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer IP is empty.";
            UTIL_THROW_EXCEPTION("socket server IP is empty.");
        }

        if (_server_port.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer port is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_s == 0) {
            MI_UTIL_LOG(MI_FATAL) << "SocketClient create INET socket failed.";
            UTIL_THROW_EXCEPTION("create INET socket failed.");
        }

        struct sockaddr_in remote;
        bzero((char*)(&remote), sizeof(remote));
        remote.sin_family = AF_INET;
        remote.sin_port = htons(atoi(_server_port.c_str()));
        memcpy((char*)(&remote.sin_addr.s_addr), _server_ip.c_str(), _server_ip.size());

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
        MI_UTIL_LOG(MI_INFO) << "INET socket connect success.\n";
    }
    
    _fd_server = fd_s;

    while (true) {
        if (!_alive) {
            break;
        }

        IPCDataHeader header;
        char* buffer = nullptr;

        if (recv(_fd_server, &header, sizeof(header) , 0) <= 0) {
            MI_UTIL_LOG(MI_WARNING) << "client receive data header failed.";
            continue;
        }
        
        MI_UTIL_LOG(MI_TRACE) << "receive data header, " << STREAM_IPCHEADER_INFO(header);   

        if (header._data_len <= 0) {
            //MI_UTIL_LOG(MI_TRACE) << "client received data buffer length less than 0.";
        } else {
            buffer = new char[header._data_len];
            if (recv(_fd_server, buffer, header._data_len, 0) <= 0) {
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

    //close socket
    close(_fd_server);
    _fd_server = 0;
    MI_UTIL_LOG(MI_TRACE) << "SocketClient("<< _path<< ")" << " running end.";
}

void SocketClient::stop() {
    _alive = false;
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
        MI_UTIL_LOG(MI_ERROR) << "send data: failed to send data header. header detail: " << STREAM_IPCHEADER_INFO(dataheader);
        return;
    }

    //send context
    if (buffer != nullptr && dataheader._data_len > 0) {
        if (-1 == send(_fd_server , buffer , dataheader._data_len , 0)) {
            MI_UTIL_LOG(MI_ERROR) << "send data: failed to send data context. header detail: " << STREAM_IPCHEADER_INFO(dataheader);
            return;
        }
    }
}

int SocketClient::post(const IPCDataHeader& post_header , char* post_data, IPCDataHeader& result_header , char*& result_data)  {
    MI_UTIL_LOG(MI_TRACE) << "SocketClient("<< _path<< ")" << " running start.";

    //create scoket
    int fd_s = 0;
    if (UNIX == _socket_type) {
        fd_s = socket(AF_UNIX , SOCK_STREAM , 0);
        if (fd_s == 0) {
            MI_UTIL_LOG(MI_FATAL) << "SocketClient create UNIX socket failed.";
            UTIL_THROW_EXCEPTION("create UNIX socket failed.");
        }
    
        struct sockaddr_un remote;
        bzero((char*)(&remote), sizeof(remote));
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
        MI_UTIL_LOG(MI_INFO) << "UNIX socket connect success.\n";
    } else if (INET == _socket_type) {
        if (_server_ip.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer IP is empty.";
            UTIL_THROW_EXCEPTION("socket server IP is empty.");
        }

        if (_server_port.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer port is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_s == 0) {
            MI_UTIL_LOG(MI_FATAL) << "SocketClient create INET socket failed.";
            UTIL_THROW_EXCEPTION("create INET socket failed.");
        }

        struct sockaddr_in remote;
        bzero((char*)(&remote), sizeof(remote));
        remote.sin_family = AF_INET;
        remote.sin_port = htons(atoi(_server_port.c_str()));
        memcpy((char*)(&remote.sin_addr.s_addr), _server_ip.c_str(), _server_ip.size());

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
        MI_UTIL_LOG(MI_INFO) << "INET socket connect success.\n";
    }
    
    _fd_server = fd_s;

    //send a message
    //send header
    if (-1 == send(_fd_server , &post_header , sizeof(post_header) , 0)) {
        MI_UTIL_LOG(MI_ERROR) << "send data: failed to send data header. header detail: " << STREAM_IPCHEADER_INFO(post_header);
        return -1;
    }

    //send context
    if (post_data != nullptr && post_header._data_len > 0) {
        if (-1 == send(_fd_server , post_data , post_header._data_len , 0)) {
            MI_UTIL_LOG(MI_ERROR) << "send data: failed to send data context. header detail: " << STREAM_IPCHEADER_INFO(post_header);
            return -1;
        }
    }

    //receive a result
    result_data = nullptr;
    if (recv(_fd_server, &result_header, sizeof(result_header) , 0) <= 0) {
        MI_UTIL_LOG(MI_WARNING) << "client receive data header failed.";
        return 0;
    }
    
    MI_UTIL_LOG(MI_TRACE) << "receive data header, " << STREAM_IPCHEADER_INFO(result_header);   

    if (result_header._data_len <= 0) {
        //MI_UTIL_LOG(MI_TRACE) << "client received data buffer length less than 0.";
    } else {
        result_data = new char[result_header._data_len];
        if (recv(_fd_server, result_data, result_header._data_len, 0) <= 0) {
            MI_UTIL_LOG(MI_WARNING) << "client receive data buffer failed.";
            return -1;
        }
    }

    //close socket
    close(_fd_server);
    _fd_server = 0;
    MI_UTIL_LOG(MI_TRACE) << "SocketClient("<< _path<< ")" << " running end.";

    return 0;
}

MED_IMG_END_NAMESPACE

#endif