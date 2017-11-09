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
#include "mi_socket_list.h"

MED_IMG_BEGIN_NAMESPACE

SocketServer::SocketServer(SocketType type):
    _socket_type(type), _fd_server(0), _alive(true), _server_port(""), 
    _max_clients(30), _client_sockets(new SocketList()) {
}

SocketServer::~SocketServer() {
    if (_fd_server != 0) {
        close(_fd_server);
        _fd_server = 0;
    }
}

void SocketServer::set_max_client(int max_clients) {
    _max_clients = max_clients;
}

void SocketServer::set_path(const std::string& path) {
    _path = path;
}

std::string SocketServer::get_path() const {
    return _path;
}

void SocketServer::set_server_address(const std::string& port) {
    _server_port = port;
}

void SocketServer::get_server_address(std::string& port) const {
    port = _server_port;
}

void SocketServer::run() {
    MI_UTIL_LOG(MI_TRACE) << "IN SocketServer run.";

    int fd_s = 0;
    if (UNIX == _socket_type) {
        if (_path.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer UNIX path is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_s == 0) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer UNIX create INET socket failed.";
            UTIL_THROW_EXCEPTION("create UNIX socket failed.");
        }

        //set master socket to allow multiple connections , 
	    //this is just a good habit, it will work without this
        int opt = 1;
        if(setsockopt(fd_s, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) < 0 ) { 
		    MI_UTIL_LOG(MI_FATAL) << "SocketServer set UNIX socket option failed.";
            UTIL_THROW_EXCEPTION("set UNIX socke t option failed.");
	    } 
    
        struct sockaddr_un serv_addr_unix;
        bzero((char*)(&serv_addr_unix), sizeof(serv_addr_unix));
        serv_addr_unix.sun_family = AF_UNIX;
        for (size_t i = 0; i < _path.size(); ++i) {
            serv_addr_unix.sun_path[i] = _path[i];
        }

        socklen_t addr_len = sizeof(serv_addr_unix);
        if (0 != bind(fd_s, (struct sockaddr*)(&serv_addr_unix), addr_len)) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer INET socket binding failed.";
            UTIL_THROW_EXCEPTION("INET socket binding failed.");
        }

    } else if (INET == _socket_type) {
        if (_server_port.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer port is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_s == 0) {
            MI_UTIL_LOG(MI_FATAL) << "SocketServer create INET socket failed.";
            UTIL_THROW_EXCEPTION("create INET socket failed.");
        }

        //set master socket to allow multiple connections , 
	    //this is just a good habit, it will work without this
        int opt = 1;
        if(setsockopt(fd_s, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) < 0 ) { 
		    MI_UTIL_LOG(MI_FATAL) << "SocketServer set UNIX socket option failed.";
            UTIL_THROW_EXCEPTION("set UNIX socket option failed.");
	    } 

        struct sockaddr_in serv_addr_inet;
        bzero((char*)(&serv_addr_inet), sizeof(serv_addr_inet));
        serv_addr_inet.sin_family = AF_INET;
        serv_addr_inet.sin_port = htons(atoi(_server_port.c_str()));
        serv_addr_inet.sin_addr.s_addr = INADDR_ANY;

        socklen_t addr_len = sizeof(serv_addr_inet);
        if (0 != bind(fd_s, (struct sockaddr*)(&serv_addr_inet), addr_len)) {
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
    
    //this thread do: accept new socket ; recv client socket request
    //accept client
    fd_set fdreads;
    while (true) {
        if(!_alive) {
            break;
        }

        int max_fd = _client_sockets->make_fd(&fdreads);
        FD_SET(_fd_server, &fdreads);
        max_fd = (std::max)(max_fd, _fd_server);
        const int activity = select(max_fd+1, &fdreads, NULL, NULL, NULL);
        
        if (activity < 0 && (errno != EINTR)) {
            //error
            MI_UTIL_LOG(MI_ERROR) << "socket read fd_set select faild.";
            continue;
        } else {
            //timeout
            MI_UTIL_LOG(MI_TRACE) << "socket read fd_set select timeout.";
            continue;
        }

        //accept client socket
        if (FD_ISSET(_fd_server, &fdreads)) {
            int new_client_socket = 0;
            socklen_t addr_len = 0;
            if (UNIX == _socket_type) {
                struct sockaddr_un client_addr_unix;
                new_client_socket = accept(_fd_server, (struct sockaddr*)(&client_addr_unix), &addr_len);
                if (new_client_socket < 0) {
                    MI_UTIL_LOG(MI_ERROR) << "server unix socket accept faild.";
                } else {
                    //log client info
                    const std::string client_path = client_addr_unix.sun_path;
                    MI_UTIL_LOG(MI_INFO) << "accept client unix socket path: " << client_path;                     
                    //add new client socket to list
                    if (-1 == _client_sockets->insert_socket(new_client_socket, client_path)) {
                        MI_UTIL_LOG(MI_ERROR) << "server socket insert new client socket faild.";
                    }
                }
            } else if (INET == _socket_type) {
                struct sockaddr_in client_addr_inet;
                new_client_socket = accept(_fd_server, (struct sockaddr*)(&client_addr_inet), &addr_len);
                if (new_client_socket < 0) {
                    MI_UTIL_LOG(MI_ERROR) << "server inet socket accept faild.";
                } else {
                    //log client info
                    const std::string client_addr = inet_ntoa(client_addr_inet.sin_addr);
                    const int port = client_addr_inet.sin_port;
                    MI_UTIL_LOG(MI_INFO) << "accept client inet socket address: " << client_addr << ":" << port;                     
                    //add new client socket to list
                    if (-1 == _client_sockets->insert_socket(new_client_socket, client_addr , port)) {
                        MI_UTIL_LOG(MI_ERROR) << "server socket insert new client socket faild.";
                    }
                }
            }           
        }

        //recv client socket
        const std::set<int>& client_sockets = _client_sockets->get_sockets();
        for (auto it = client_sockets.begin(); it != client_sockets.end(); ++it) {
            const int cs = *it;
            std::string addr;
            int port(0);
            if (0 != _client_sockets->get_socket_addr(cs, addr, port)) {
                addr = "unknown";
                port = 0;
            }

            if (FD_ISSET(cs, &fdreads)) {
                //reveive dataheader
                IPCDataHeader header;
                int err = recv(cs, &header, sizeof(header), 0);
                header._receiver = cs;//very important
                if (err == 0) {
                    //client disconnect
                    if (UNIX == _socket_type) {
                        MI_UTIL_LOG(MI_INFO) << "client unix socket path:" << addr << " disconnect.";
                    } else if(INET == _socket_type) {
                        MI_UTIL_LOG(MI_INFO) << "client inet socket address:" << addr << ":" << port << " disconnect.";
                    }
                    close(cs);
                    _client_sockets->remove_socket(cs);
                    this->clear_package_i(cs);
                    continue;    
                } else if (err < 0) {
                    //recv failed
                    if (UNIX == _socket_type) {
                        MI_UTIL_LOG(MI_INFO) << "client unix socket path:" << addr << " recv failed.";
                    } else if(INET == _socket_type) {
                        MI_UTIL_LOG(MI_INFO) << "client inet socket address:" << addr << ":" << port << " recv failed.";
                    }
                    close(cs);
                    _client_sockets->remove_socket(cs);
                    this->clear_package_i(cs);
                    continue;  
                }

                //receive data buffer
                if (header._data_len <= 0) {
                    //MI_UTIL_LOG(MI_TRACE) << "server received data buffer length less than 0.";
                } else {
                    char* buffer = new char[header._data_len]; 
                    err = recv(_fd_server, buffer, header._data_len, 0);
                    if (err == 0) {
                        //client disconnect
                          if (UNIX == _socket_type) {
                            MI_UTIL_LOG(MI_INFO) << "client unix socket path:" << addr << " disconnect.";
                            MI_UTIL_LOG(MI_WARNING) << "client unix socket send data damaged.";
                        } else if(INET == _socket_type) {
                            MI_UTIL_LOG(MI_INFO) << "client inet socket address:" << addr << ":" << port << " disconnect.";
                            MI_UTIL_LOG(MI_WARNING) << "client inet socket send data damaged.";
                        }
                        //close socket
                        close(cs);
                        _client_sockets->remove_socket(cs);
                        this->clear_package_i(cs);
                        continue;
                    } else if (err < 0) {
                        //recv failed
                        if (UNIX == _socket_type) {
                            MI_UTIL_LOG(MI_INFO) << "client unix socket path:" << addr << " recv failed.";
                        } else if(INET == _socket_type) {
                            MI_UTIL_LOG(MI_INFO) << "client inet socket address:" << addr << ":" << port << " recv failed.";
                        }
                        close(cs);
                        _client_sockets->remove_socket(cs);
                        this->clear_package_i(cs);
                        continue;  
                    }

                    //add client socket fd to header
                    header._receiver = (unsigned int)cs;

                    try {
                        if (_handler) {
                            _handler->handle(header, buffer);
                        } else {
                            MI_UTIL_LOG(MI_WARNING) << "client handler to process received data is null.";
                        }
                    } catch(const Exception& e) {
                        //Ignore error to keep connecting
                        MI_UTIL_LOG(MI_FATAL) << "handle command error(skip and continue): " << e.what();
                    }
                }
            }            
        }
    }

    //close socket
    close(_fd_server);
    _fd_server = 0;
    
    MI_UTIL_LOG(MI_TRACE) << "OUT SocketServer run.";
}


void SocketServer::stop() {
    _alive = false;
}

void SocketServer::register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler) {
    _handler = handler;
}

void SocketServer::send_data(const IPCDataHeader& dataheader , char* buffer) {
    //this thread gather package to be sending
    Package *pkg = new Package(dataheader, buffer);
    const int client_socket_fd = (int)(dataheader._receiver);
    push_back_package_i(client_socket_fd, pkg);
}

int SocketServer::send() {
    try_pop_front_package_i();

    //this thread send data to clients
    fd_set fdreads;
    if(_client_sockets->get_socket_count() == 0) {
        return 0;
    }

    FD_ZERO(&fdreads);
    int max_fd = _client_sockets->make_fd(&fdreads);
    const int activity = select(max_fd+1, &fdreads, NULL, NULL, NULL);

    if (activity < 0 && (errno != EINTR)) {
        //error
        MI_UTIL_LOG(MI_ERROR) << "socket read fd_set select faild.";
        return activity;
    } else {
        //timeout
        MI_UTIL_LOG(MI_TRACE) << "socket read fd_set select timeout.";
        return activity;
    }

    const std::set<int>& client_sockets = _client_sockets->get_sockets();
    for (auto it = client_sockets.begin(); it != client_sockets.end(); ++it) {
        const int cs = *it;
        std::string addr;
        int port(0);
        if (0 != _client_sockets->get_socket_addr(cs, addr, port)) {
            addr = "unknown";
            port = 0;
        }

        if (FD_ISSET(cs, &fdreads)) {
            //check package
            Package* pkg_send = nullptr;
            {
                boost::mutex::scoped_lock locker(_mutex_package);
                auto client_store = _client_pkg_store.find(cs);
                if (client_store == _client_pkg_store.end()) {
                    continue;
                } 
                if (client_store->second.size() == 0) {
                    continue;
                }
                pkg_send = client_store->second.front();
                client_store->second.pop_front();
            }

            //check close
            if (-1 == _client_sockets->check_socket(cs)) {
                //socket has been close
                //drop package 
                if (nullptr != pkg_send) {
                    delete pkg_send;
                    pkg_send = nullptr;
                }
                continue;
            }
            if (-1 == ::send(cs, &(pkg_send->header), sizeof(pkg_send->header), 0)) {
                MI_UTIL_LOG(MI_ERROR) << "send data: failed to send data header. header detail: " 
                << STREAM_IPCHEADER_INFO(pkg_send->header);
                continue;
            }

            if (nullptr != pkg_send->buffer && pkg_send->header._data_len > 0) {
                if (-1 == ::send(cs , pkg_send->buffer , pkg_send->header._data_len , 0)) {
                    MI_UTIL_LOG(MI_ERROR) << "send data: failed to send data context. header detail: " 
                    << STREAM_IPCHEADER_INFO(pkg_send->header);
                }
            }
        }            
    }

    return 0;
}

void SocketServer::push_back_package_i(int socket_fd, Package* pkg) {
    boost::mutex::scoped_lock locker(_mutex_package);
    auto client_store = _client_pkg_store.find(socket_fd);
    if (client_store == _client_pkg_store.end()) {
        PackageStore pkg_store;
        pkg_store.push_back(pkg);
        _client_pkg_store.insert(std::make_pair(socket_fd, pkg_store));
    } else {
        client_store->second.push_back(pkg);
    }
    _condition_empty_package.notify_one();
}

SocketServer::Package* SocketServer::pop_front_package_i(int socket_fd) {
    boost::mutex::scoped_lock locker(_mutex_package);
    auto client_store = _client_pkg_store.find(socket_fd);
    if (client_store == _client_pkg_store.end()) {
        return nullptr;
    } 
    if (client_store->second.size() == 0) {
        return nullptr;
    }
    Package* pkg = client_store->second.front();
    client_store->second.pop_front();
    return pkg;
}

void SocketServer::clear_package_i(int socket_fd) {
    PackageStore old_package_store;
    {
        boost::mutex::scoped_lock locker(_mutex_package);
        auto client_store = _client_pkg_store.find(socket_fd);
        if (client_store != _client_pkg_store.end()) {
            old_package_store = client_store->second;
            _client_pkg_store.erase(client_store);
        }   
    }

    while (old_package_store.empty()) {
        Package* pkg = old_package_store.front();
        if (nullptr != pkg) {
            delete pkg;
            pkg = nullptr; 
        }
        old_package_store.pop_front();
    }
}

void SocketServer::try_pop_front_package_i() {
    boost::mutex::scoped_lock locker(_mutex_package);
    while(_client_pkg_store.empty()) {
        _condition_empty_package.wait(_mutex_package);
    }
}


MED_IMG_END_NAMESPACE