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
    _socket_type(type),_path(""),_server_ip("127.0.0.1"),_server_port("8888"), _fd_server(0) {

}

SocketClient::~SocketClient() {
    this->stop();
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

void SocketClient::connect() {
    _server_info = (UNIX == _socket_type) ? ("{UNIX:"+_path+"}") : ("{INET:"+_server_ip+":"+_server_port+"}");

    int fd_s = 0;
    if (UNIX == _socket_type) {
        fd_s = socket(AF_UNIX , SOCK_STREAM , 0);
        if (fd_s <= 0) {
            MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " create socket failed. errno: " << errno;
            UTIL_THROW_EXCEPTION("create socket failed.");
        }
    
        struct sockaddr_un remote;
        bzero((char*)(&remote), sizeof(remote));
        remote.sun_family = AF_UNIX;
        for (size_t i = 0; i < _path.size(); ++i) {
            remote.sun_path[i] = _path[i];
        }
        socklen_t len = sizeof(remote);

        //connect 100 times once per second
        int connect_status = -1;
        for (int i = 0; i < _reconnect_times; ++i) {
            connect_status = ::connect(fd_s , (struct sockaddr*)(&remote) , len);
            if (-1 != connect_status) {
                break;
            }
            MI_UTIL_LOG(MI_INFO) << "trying connect to server times(once per second): " << i;
            sleep(1);
        }
    
        if (-1 == connect_status) {
            MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " connect failed. errno: " << errno;
            UTIL_THROW_EXCEPTION("connect server failed.");
        }
    } else if (INET == _socket_type) {
        if (_server_ip.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " ip is empty.";
            UTIL_THROW_EXCEPTION("socket server ip is empty.");
        }

        if (_server_port.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " port is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_s <= 0) {
            MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " create socket failed. errno: " << errno;
            UTIL_THROW_EXCEPTION("create socket failed.");
        }

        struct sockaddr_in remote;
        bzero((char*)(&remote), sizeof(remote));
        remote.sin_family = AF_INET;
        remote.sin_port = htons(atoi(_server_port.c_str()));
        struct hostent *server = gethostbyname(_server_ip.c_str());
        if (server == NULL) {
            MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " no such hostname.";
            UTIL_THROW_EXCEPTION("socket client no such hostname.");
        }
        bcopy((char *)server->h_addr, (char *)&remote.sin_addr.s_addr, server->h_length);
        socklen_t len = sizeof(remote);

        //connect 100 times once per second
        int connect_status = -1;
        for (int i = 0; i < _reconnect_times; ++i) {
            connect_status = ::connect(fd_s, (struct sockaddr*)(&remote), len);
            if (connect_status != -1) {
                break;
            }
            MI_UTIL_LOG(MI_INFO) << "trying connect to server times(once per second): " << i;
            sleep(1);
        }
    
        if (-1 == connect_status) {
            MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " connect failed. errno: " << errno;
            UTIL_THROW_EXCEPTION("connect server failed.");
        }
    }

    MI_UTIL_LOG(MI_INFO) << "client " << _server_info << " connect success.";
    _fd_server = fd_s;
}

void SocketClient::run() {
    MI_UTIL_LOG(MI_TRACE) << "IN SocketClient run.";

    connect();

    //execute on connect event
    if (_fd_server != 0 && _on_connect_event) {
        _on_connect_event->execute();
    }

    while (true) {
        //------------------//
        //1 recv header
        //------------------//
        IPCDataHeader header;
        //header is just 32 byte,use MSG_WAITALL to force client socket to return untill recv all header buffer(32byte)
        const int header_size = recv(_fd_server, &header, sizeof(header) , MSG_WAITALL);
        if (header_size < 0) {
            //socket error 
            MI_UTIL_LOG(MI_ERROR) << "client " << _server_info << " receive data header failed. errno: " << errno;
            break;
        }
        if (header_size == 0 && _fd_server == 0) {
            //closed by other thread
            MI_UTIL_LOG(MI_INFO) << "client " << _server_info << " quit because socket is closed by other thread.";
            break;
        }
        if (header_size == 0) {
            //server disconnected
            MI_UTIL_LOG(MI_WARNING) << "client " << _server_info <<  " quit because server close the connect.";
            break;
        }  

        //-----------------------//
        //2 recv data
        //-----------------------//
        char* buffer = nullptr;
        if (header.data_len > 0) {
            buffer = new char[header.data_len];
            int cur_size = 0;
            int accum_size = 0;
            int try_size = (int)header.data_len;
            bool quit_signal = false;
            while (accum_size < (int)header.data_len) {
                cur_size = recv(_fd_server, buffer+accum_size, try_size, 0);
                if (cur_size < 0) {
                    //socket error 
                    MI_UTIL_LOG(MI_ERROR) << "client " << _server_info << " recv data failed.";
                    delete [] buffer;
                    buffer = nullptr;
                    quit_signal = true;
                    break;
                } else if (cur_size == 0 && _fd_server == 0) {
                    //closed by other thread
                    MI_UTIL_LOG(MI_INFO) << "client " << _server_info << " quit because socket is closed by other thread.";
                    delete [] buffer;
                    buffer = nullptr;
                    quit_signal = true;
                    break;
                } else if (cur_size == 0) {
                    //server disconnected
                    MI_UTIL_LOG(MI_WARNING) << "client " << _server_info << " quit because server close the connect.";
                    delete [] buffer;
                    buffer = nullptr;
                    quit_signal = true;
                    break;
                }
                accum_size += cur_size;
                try_size -= cur_size;
            }
            
            if (quit_signal) {
                break;
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
            MI_UTIL_LOG(MI_ERROR) << "handle command error(skip and continue): " << e.what();
        }
    }

    //close socket
    if (0 != _fd_server) {
        close(_fd_server);   
        _fd_server = 0;
    }
    
    MI_UTIL_LOG(MI_TRACE) << "OUT SocketClient run.";
}

void SocketClient::stop() {
    if (0 != _fd_server) {
        close(_fd_server);
        _fd_server = 0;
    }
}

int SocketClient::sync_send_data(const IPCDataHeader& dataheader , char* buffer) {
    if (-1 == _fd_server) {
        MI_UTIL_LOG(MI_FATAL) << "client " << _server_info << " send data failed with invalid server fd.";
        return -1;
    }

    //------------------//
    //1 send header
    //------------------//
    if (-1 == send(_fd_server , &dataheader , sizeof(dataheader) , 0)) {
        MI_UTIL_LOG(MI_ERROR) << "client " << _server_info << " send data header failed with errno: " << errno << 
        ". header detail: " << STREAM_IPCHEADER_INFO(dataheader);
        return -1;
    }

    //------------------//
    //1 send data
    //------------------//
    if (buffer != nullptr && dataheader.data_len > 0) {
        if (-1 == send(_fd_server , buffer , dataheader.data_len , 0)) {
            MI_UTIL_LOG(MI_ERROR) << "client " << _server_info << " send data failed with errno: " << errno << 
            ". header detail: " << STREAM_IPCHEADER_INFO(dataheader);
            return -1;
        }
    }
    return 0;
}

int SocketClient::sync_send_data(IPCPackage* package) {
    if (package) {
        return this->sync_send_data(package->header, package->buffer);
    } else {
        return -1;
    }
}

int SocketClient::sync_send_data(const std::vector<IPCPackage*>& packages) {
    int err = 0;
    for(auto it = packages.begin(); it != packages.end() ; ++it) {
        IPCPackage *pkg = *it;
        const IPCDataHeader& post_header = pkg->header;
        char* post_data = pkg->buffer;
        err += this->sync_send_data(post_header, post_data);
    }
    if (err == 0) {
        return 0;
    } else {
        return -1;
    }
}

int SocketClient::sync_post(const std::vector<IPCPackage*>& packages) {
    MI_UTIL_LOG(MI_TRACE) << "IN SocketClient post.";

    connect();

    sync_send_data(packages);

    //recv result
    while (true) {
        //------------------//
        //1 recv header
        //------------------//
        IPCDataHeader header;
        //header is just 32 byte,use MSG_WAITALL to force client socket to return untill recv all header buffer(32byte) 
        const int header_size = recv(_fd_server, &header, sizeof(header) , MSG_WAITALL);
        if (header_size < 0) {
            //socket error 
            MI_UTIL_LOG(MI_ERROR) << "client " << _server_info << " receive data header failed. errno: " << errno;
            break;
        }
        if (header_size == 0 && _fd_server == 0) {
            //closed by other thread
            MI_UTIL_LOG(MI_INFO) << "client " << _server_info << " quit because socket is closed by other thread.";
            break;
        }
        if (header_size == 0) {
            //server disconnected
            MI_UTIL_LOG(MI_WARNING) << "client " << _server_info <<  " quit because server close the connect.";
            break;
        }

        //------------------//
        //2 recv data
        //------------------//
        char* buffer = nullptr;
        if (header.data_len > 0) {
            buffer = new char[header.data_len];
            int cur_size = 0;
            int accum_size = 0;
            int try_size = (int)header.data_len;
            bool quit_signal = false;
            while (accum_size < (int)header.data_len) {
                cur_size = recv(_fd_server, buffer+accum_size, try_size, 0);
                if (cur_size < 0) {
                    //socket error 
                    MI_UTIL_LOG(MI_ERROR) << "client " << _server_info << " recv data failed.";
                    delete [] buffer;
                    buffer = nullptr;
                    quit_signal = true;
                    break;
                } else if (cur_size == 0 && _fd_server == 0) {
                    //closed by other thread
                    MI_UTIL_LOG(MI_INFO) << "client " << _server_info << " quit because socket is closed by other thread.";
                    delete [] buffer;
                    buffer = nullptr;
                    quit_signal = true;
                    break;
                } else if (cur_size == 0) {
                    //server disconnected
                    MI_UTIL_LOG(MI_WARNING) << "client " << _server_info << " quit because server close the connect.";
                    delete [] buffer;
                    buffer = nullptr;
                    quit_signal = true;
                    break;
                }
                accum_size += cur_size;
                try_size -= cur_size;
            }
            
            if (quit_signal) {
                break;
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
            return -1;
        }
    }

    //close socket
    if (0 != _fd_server) {
        close(_fd_server);
        _fd_server = 0;
    }

    MI_UTIL_LOG(MI_INFO) << "client " << _server_info << " post data done.\n";

    MI_UTIL_LOG(MI_TRACE) << "OUT SocketClient post.";;

    return 0;
}

void SocketClient::on_connect(std::shared_ptr<IEvent> ev) {
    _on_connect_event = ev;
}

MED_IMG_END_NAMESPACE

#endif