#include "mi_socket_server.h"

#include <stdio.h> 
#include <string.h> //strlen 
#include <stdlib.h> 
#include <errno.h> 
#include <unistd.h> //close 
#include <arpa/inet.h> //close 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <sys/un.h>
#include <netinet/in.h> 
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros 
    
#include "mi_exception.h"
#include "mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

SocketServer::SocketServer(SocketType type):
    _socket_type(type), _fd_server(-1), _alive(true), _server_port(""), _max_clients(30) {
}

SocketServer::~SocketServer() {
    if (_fd_server != -1) {
        close(_fd_server);
        _fd_server = -1;
    }
}

void SocketServer::set_max_client(int max_clients) {
    _max_clients = max_clients;
}

void SocketServer::set_server_address(const std::string& port) {
    _server_port = port;
}

void SocketServer::get_server_address(std::string& port) const {
    port = _server_port;
}

void SocketServer::run() {
    MI_UTIL_LOG(MI_TRACE) << "IN SocketServer run.";

    int fd_s = -1;
    if (UNIX == _socket_type) {
        if (_path.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer UNIX path is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_s == -1) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer UNIX create INET socket failed.";
            UTIL_THROW_EXCEPTION("create UNIX socket failed.");
        }

        int opt = 1;
        if(setsockopt(fd_s, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) < 0 ) { 
		    MI_UTIL_LOG(MI_FATAL) << "SocketServer set UNIX socket option failed.";
            UTIL_THROW_EXCEPTION("set UNIX socket option failed.");
	    } 
    
        struct sockaddr_un serv_addr;
        bzero((char*)(&serv_addr), sizeof(serv_addr));

        serv_addr.sun_family = AF_UNIX;
        for (size_t i = 0; i < _path.size(); ++i) {
            serv_addr.sun_path[i] = _path[i];
        }

        socklen_t len = sizeof(serv_addr);
        if (0 != bind(fd_s, (struct sockaddr*)(&serv_addr), len)) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer INET socket binding failed.";
            UTIL_THROW_EXCEPTION("INET socket binding failed.");
        }

    } else if (INET == _socket_type) {
        if (_server_port.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer port is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_s == -1) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer create INET socket failed.";
            UTIL_THROW_EXCEPTION("create INET socket failed.");
        }

        int opt = 1;
        if(setsockopt(fd_s, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) < 0 ) { 
		    MI_UTIL_LOG(MI_FATAL) << "SocketServer set UNIX socket option failed.";
            UTIL_THROW_EXCEPTION("set UNIX socket option failed.");
	    } 

        struct sockaddr_in serv_addr;
        bzero((char*)(&serv_addr), sizeof(serv_addr));

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(atoi(_server_port.c_str()));
        serv_addr.sin_addr.s_addr = INADDR_ANY;

        socklen_t len = sizeof(serv_addr);
        if (0 != bind(fd_s, (struct sockaddr*)(&serv_addr), len)) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer INET socket binding failed.";
            UTIL_THROW_EXCEPTION("INET socket binding failed.");
        }
    }
    _fd_server = fd_s;

    //try to specify maximum of 3 pending connections for the master socket 
	if (listen(_fd_server, 3) < 0) { 
		MI_UTIL_LOG(MI_FATAL) << "SocketServer listen failed.";
        UTIL_THROW_EXCEPTION("socket listen failed.");
    } 
    
    //accept client
    while (true) {
        if(!_alive) {
            break;
        }

        //set of socket descriptors 
	    fd_set readfds;


    }

    MI_UTIL_LOG(MI_TRACE) << "OUT SocketServer run.";
}


void SocketServer::stop() {
    _alive = false;
}

void SocketServer::send_data(const IPCDataHeader& dataheader , char* buffer) {
    //TODO lock when io op
    Package *pkg = new Package(dataheader, buffer);
    auto client_store = _client_pkg_store.find((int)(dataheader._receiver));
    if (client_store == _client_pkg_store.end()) {
        MI_UTIL_LOG(MI_ERROR) << "can't find receiver socket when sending data.";
    } else {
        client_store->second.push_back(pkg);
    }
}


MED_IMG_END_NAMESPACE