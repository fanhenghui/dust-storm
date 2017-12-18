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
#include "mi_memory_shield.h"

MED_IMG_BEGIN_NAMESPACE

SocketServer::SocketServer(SocketType type):
    _socket_type(type),_path(""),_server_port(""),_fd_server(0),_max_clients(30),_alive(true),  
    _client_sockets(new SocketList()),_package_cache_capcity(1024.0f*10.0f),_package_cache_size(0.0f) {
}

SocketServer::~SocketServer() {
    if (_fd_server != 0) {
        shutdown(_fd_server, SHUT_RDWR);
        _fd_server = 0;
        MI_UTIL_LOG(MI_INFO) << "close socket when destruction.";
    }
    if (!_path.empty()) {
        unlink(_path.c_str());
        MI_UTIL_LOG(MI_INFO) << "unlink UNIX path when destruction.";
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
    _server_info = (UNIX == _socket_type) ? ("{UNIX:"+_path+"}") : ("{INET:"+_server_port+"}");
    
    int fd_s = 0;
    if (UNIX == _socket_type) {
        if (_path.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " UNIX path is empty.";
            UTIL_THROW_EXCEPTION("socket server unix path is empty.");
        }

        fd_s = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_s <= 0) {
            MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " create socket failed. errno: " << errno;
            UTIL_THROW_EXCEPTION("create socket failed.");
        }

        //set master to allow multiple connections , and bind closed socket immediately 
        int opt = 1;
        if(setsockopt(fd_s, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) < 0 ) { 
		    MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " set socket option failed.";
            UTIL_THROW_EXCEPTION("set socket option failed.");
	    } 
    
        struct sockaddr_un serv_addr_unix;
        bzero((char*)(&serv_addr_unix), sizeof(serv_addr_unix));
        serv_addr_unix.sun_family = AF_UNIX;
        for (size_t i = 0; i < _path.size(); ++i) {
            serv_addr_unix.sun_path[i] = _path[i];
        }

        socklen_t addr_len = sizeof(serv_addr_unix);
        
        if (0 != bind(fd_s, (struct sockaddr*)(&serv_addr_unix), addr_len)) {
            shutdown(fd_s, SHUT_RDWR);
            MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " binding failed. errno: " << errno;
            _path = "";//clean path to prevent unlink it(others is using it.)
            UTIL_THROW_EXCEPTION("socket binding failed.");
        }

    } else if (INET == _socket_type) {
        if (_server_port.empty()) {
            MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " port is empty.";
            UTIL_THROW_EXCEPTION("socket server port is empty.");
        }

        fd_s = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_s <= 0) {
            MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " create socket failed. errno: " << errno;
            UTIL_THROW_EXCEPTION("create socket failed.");
        }

        //set master to allow multiple connections , and bind closed socket immediately 
        int opt = 1;
        if(setsockopt(fd_s, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt)) < 0 ) { 
		    MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " set socket option failed.";
            UTIL_THROW_EXCEPTION("set socket option failed.");
	    } 

        struct sockaddr_in serv_addr_inet;
        bzero((char*)(&serv_addr_inet), sizeof(serv_addr_inet));
        serv_addr_inet.sin_family = AF_INET;
        serv_addr_inet.sin_port = htons(atoi(_server_port.c_str()));
        serv_addr_inet.sin_addr.s_addr = INADDR_ANY;

        socklen_t addr_len = sizeof(serv_addr_inet);
        if (0 != bind(fd_s, (struct sockaddr*)(&serv_addr_inet), addr_len)) {
            shutdown(fd_s, SHUT_RDWR);
            MI_UTIL_LOG(MI_FATAL) << "server " << _server_info <<  " binding failed. errno: " << errno; ;
            UTIL_THROW_EXCEPTION("socket binding failed.");
        }
    }
    _fd_server = fd_s;

    //try to specify maximum of 3 pending connections for the master socket 
	if (0 != listen(_fd_server, 20)) { 
		MI_UTIL_LOG(MI_FATAL) << "server " << _server_info << " listen failed. errno:" << errno;
        UTIL_THROW_EXCEPTION("socket listen failed.");
    } 

    //---------------------------------------//
    //below code don't use exception to break
    //---------------------------------------//

    //accept client
    fd_set fdreads;
    while (true) {
        if(!_alive) {
            break;
        }

        FD_ZERO(&fdreads);
        FD_SET(_fd_server, &fdreads);

        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 500000;
        const int activity = select(_fd_server+1, &fdreads, NULL, NULL, &timeout);
        if (activity < 0 ) {
            if (errno == EINTR) {
                //timeout
                continue;
            } else {
                MI_UTIL_LOG(MI_ERROR) << "server " << _server_info <<  " read fd_set select faild. errno: " << errno;
                continue;
            }
        } else if (activity == 0) {
            //0 to read
            continue;} 

        //accept client socket
        if (FD_ISSET(_fd_server, &fdreads)) {
            int new_client_socket = 0;
            
            if (UNIX == _socket_type) {
                socklen_t addr_len = sizeof(sockaddr_un);
                struct sockaddr_un client_addr_unix;
                new_client_socket = accept(_fd_server, (struct sockaddr*)(&client_addr_unix), &addr_len);
                if (new_client_socket < 0) {
                    MI_UTIL_LOG(MI_ERROR) << "server " << _server_info << " accept failed. errno: " << errno;
                    continue;
                }

                const std::string client_path = client_addr_unix.sun_path;
                MI_UTIL_LOG(MI_INFO) << "server " << _server_info << "accept client socket path: " << client_path;                     
                if (-1 == _client_sockets->insert_socket(new_client_socket, client_path)) {
                    MI_UTIL_LOG(MI_ERROR) << "server " << _server_info << " insert new client socket faild.";
                }
                continue;
            }
            
            if (INET == _socket_type) {
                socklen_t addr_len = sizeof(sockaddr_in);
                struct sockaddr_in client_addr_inet;
                new_client_socket = accept(_fd_server, (struct sockaddr*)(&client_addr_inet), &addr_len);
                if (new_client_socket < 0) {
                   MI_UTIL_LOG(MI_ERROR) << "server " << _server_info << " accept failed. errno: " << errno;
                   continue;
                }

                const std::string client_addr = inet_ntoa(client_addr_inet.sin_addr);
                const int port = client_addr_inet.sin_port;
                MI_UTIL_LOG(MI_INFO) << "server " << _server_info << " accept client socket address: " << client_addr << ":" << port;                     
                if (-1 == _client_sockets->insert_socket(new_client_socket, client_addr, port)) {
                    MI_UTIL_LOG(MI_ERROR) << "server " << _server_info << " insert new client socket faild.";
                }
                continue;
            }           
        }
        //loop end
    }

    //close socket
    shutdown(_fd_server, SHUT_RDWR);
    _fd_server = 0;
    MI_UTIL_LOG(MI_INFO) << "server " << _server_info << " close success.";
    
    if (_socket_type == UNIX) {
        unlink(_path.c_str());
        MI_UTIL_LOG(MI_INFO) << "server socket unlink UNIX path: " << _path;
        _path = "";
    }
    
    MI_UTIL_LOG(MI_TRACE) << "OUT SocketServer run.";
}

void SocketServer::stop() {
    _alive = false;
}

int SocketServer::recv() {
    //waiting for fd_server created
    if (0 == _fd_server) {
        while(_alive) {
            sleep(1);
            if (0 != _fd_server) {
                break;
            }
            MI_UTIL_LOG(MI_DEBUG) << "wait to connect: " << _path;
        }
    }
    
    fd_set fdreads;
    int max_fd = _client_sockets->make_fd(&fdreads);
    if (max_fd == 0) {
        //没有client 接入的情况：
        //方案1：直接sleep 然后 return，不进入select.
        //方案2（目前采用）：调用select，会超时返回表示没有任何可读的fd（注意select一定要是non-block）
    }

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 500000; //5ms
    const int activity = select(max_fd+1, &fdreads, NULL, NULL, &timeout);
    if (activity < 0 ) {
        if (errno == EINTR) {
            //timeout
            return 0;
        } else {
            MI_UTIL_LOG(MI_ERROR) << "server " << _server_info <<  " read fd_set select faild. error: " << errno;
            return -1;
        }
    } else if (activity == 0) {
        //0 to read
        return 0;
    }

    //recv client socket
    const std::map<unsigned int, SocketList::SocketInfo> client_socket_infos  = _client_sockets->get_socket_infos(); 
    for (auto it = client_socket_infos.begin(); it != client_socket_infos.end(); ++it) {
        const unsigned int socket_id = it->first;
        const int socket = it->second.socket;
        const std::string host = it->second.host;
        const int port = it->second.port;
        const unsigned int time = it->second.time;
        if (FD_ISSET(socket, &fdreads)) {
            //-----------------------//
            //1 recv header
            //-----------------------//
            IPCDataHeader header;
            //header is just 32 byte,use MSG_WAITALL to force client socket to return untill recv all header buffer(32byte) 
            int err = ::recv(socket, &header, sizeof(header) , MSG_WAITALL);

            //set client socket id/created_time to header to make operation know which socket to interact
            header.receiver = socket_id;//client socket id
            header.reserved1 = time;//client socket created time

            //client disconnect
            if (err == 0) {
                if (UNIX == _socket_type) {
                    MI_UTIL_LOG(MI_INFO) << "client unix socket path:" << host << " disconnect.";
                } else if(INET == _socket_type) {
                    MI_UTIL_LOG(MI_INFO) << "client inet socket address:" << host << ":" << port << " disconnect.";
                }
                shutdown(socket, SHUT_RDWR);
                _client_sockets->remove_socket(socket_id);
                this->clear_package(socket_id);
                continue;    
            } 
            
            //recv error
            if (err < 0) {
                if (UNIX == _socket_type) {
                    MI_UTIL_LOG(MI_ERROR) << "client unix socket path:" << host << " recv failed. errno: " << errno;
                } else if(INET == _socket_type) {
                    MI_UTIL_LOG(MI_ERROR) << "client inet socket address:" << host << ":" << port << " recv failed. errno: " << errno;
                }
                //close this client
                shutdown(socket, SHUT_RDWR);
                _client_sockets->remove_socket(socket_id);
                this->clear_package(socket_id);
                continue;  
            }

            //-----------------------//
            //2 recv data
            //-----------------------//
            char* buffer = nullptr;
            if (header.data_len > 0) {
                //loop to recv data
                buffer = new char[header.data_len];
                int cur_size = 0;
                int accum_size = 0;
                int try_size = (int)header.data_len;
                while (accum_size < (int)header.data_len) {
                    cur_size = ::recv(socket, buffer+accum_size, try_size, 0);
                    if (cur_size <= 0) {
                        MI_UTIL_LOG(MI_ERROR) << "client receive data buffer failed.";
                        break;
                    }
                    accum_size += cur_size;
                    try_size -= cur_size;
                }

                //client disconnect
                if (cur_size == 0) {
                    if (UNIX == _socket_type) {
                        MI_UTIL_LOG(MI_INFO) << "client unix socket path:" << host << " disconnect.";
                    } else if(INET == _socket_type) {
                        MI_UTIL_LOG(MI_INFO) << "client inet socket address:" << host << ":" << port << " disconnect.";
                    }

                    delete [] buffer;
                    buffer = nullptr;
                    shutdown(socket, SHUT_RDWR);
                    _client_sockets->remove_socket(socket_id);
                    this->clear_package(socket_id);
                    continue;
                }
                
                //recv error
                if (cur_size < 0) {
                    if (UNIX == _socket_type) {
                        MI_UTIL_LOG(MI_ERROR) << "client unix socket path:" << host << " recv failed.";
                    } else if(INET == _socket_type) {
                        MI_UTIL_LOG(MI_ERROR) << "client inet socket address:" << host << ":" << port << " recv failed.";
                    }

                    delete [] buffer;
                    buffer = nullptr;
                    shutdown(socket, SHUT_RDWR);
                    _client_sockets->remove_socket(socket_id);
                    this->clear_package(socket_id);
                    continue;  
                }
            }

            //MI_UTIL_LOG(MI_DEBUG) << "server " << _server_info << " recv package: {MsgID: " << header.msg_id << "; OpID: " << header.op_id << "}";

            try {
                if (_handler) {
                    _handler->handle(header, buffer);
                } else {
                    MI_UTIL_LOG(MI_ERROR) << "server handler to process received data is null.";
                }
            } catch(const Exception& e) {
                //Ignore error to keep connecting
                MI_UTIL_LOG(MI_FATAL) << "server handle command error(skip and continue): " << e.what();
            }
        }            
    }    

    return 0;
}

void SocketServer::register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler) {
    _handler = handler;
}

int SocketServer::async_send_data(IPCPackage* package) {
    //this thread gather package to be sending
    const unsigned int socket_list_id = package->header.receiver;
    if (-1 == _client_sockets->check_socket(socket_list_id)) {
        //MI_UTIL_LOG(MI_DEBUG) << "async send data failed, invalid socket list id : "  << socket_list_id;
        return -1;
    } else {
        push_back_package(socket_list_id, package);
        //MI_UTIL_LOG(MI_DEBUG) << "async send data success, socket list id : "  << socket_list_id;
        return 0;
    }
}

int SocketServer::send() {
    try_pop_front_package();

    //this thread send data to clients
    fd_set fdwrites;
    if(_client_sockets->get_socket_count() == 0) {
        return 0;
    }

    FD_ZERO(&fdwrites);
    int max_fd = _client_sockets->make_fd(&fdwrites);
    if (max_fd == 0) { //double check
        return 0;
    }

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 500000; //5ms
    const int activity = select(max_fd+1, NULL, &fdwrites, NULL, &timeout);
    if (activity < 0 ) {
        if (errno == EINTR) {
            //timeout
            return 0;
        } else {
            MI_UTIL_LOG(MI_ERROR) << "server " << _server_info <<  "client write fd_set select faild. error: " << errno;
            return -1;
        }
    } else if (activity == 0) {
        //0 to write
        return 0;
    }

    const std::map<unsigned int, SocketList::SocketInfo> client_socket_infos = _client_sockets->get_socket_infos();
    for (auto it = client_socket_infos.begin(); it != client_socket_infos.end(); ++it) {
        const unsigned int socket_id = it->first;
        const int socket = it->second.socket;
        const std::string host = it->second.host;
        //const int port = it->second.port;
        //const unsigned int time = it->second.time;

        if (FD_ISSET(socket, &fdwrites)) {
            //check package
            IPCPackage* pkg_send = pop_front_package(socket_id);
            StructShield<IPCPackage> shield(pkg_send);
            if(nullptr == pkg_send) {
                //pop empty package
                continue;
            }

            //check valid each client socket (may remove in recv(cmd_handler) thread)
            if (-1 == _client_sockets->check_socket(socket_id)) {
                //socket has been close, drop it 
                continue;
            }

            //------------------//
            //send header
            //------------------//
            if (-1 == ::send(socket, &(pkg_send->header), sizeof(pkg_send->header), 0)) {
                MI_UTIL_LOG(MI_ERROR) << "server " << _server_info << " send data header failed with errno: " << errno << 
                ". header detail: " << STREAM_IPCHEADER_INFO(pkg_send->header);
                continue;
            }

            //------------------//
            //send data
            //------------------//
            if (nullptr != pkg_send->buffer && pkg_send->header.data_len > 0) {
                if (-1 == ::send(socket , pkg_send->buffer , pkg_send->header.data_len , 0)) {
                    MI_UTIL_LOG(MI_ERROR) << "server " << _server_info << " send data failed with errno: " << errno << 
                    ". header detail: " << STREAM_IPCHEADER_INFO(pkg_send->header);   
                }
            }
        }            
    }

    return 0;
}

void SocketServer::push_back_package(unsigned int socket_list_id, IPCPackage* pkg) {
    boost::mutex::scoped_lock locker(_mutex_package);
    while (package_cache_full(pkg)) {
        _condition_full_package.wait(_mutex_package);
    }

    auto client_store = _client_pkg_store.find(socket_list_id);
    if (client_store == _client_pkg_store.end()) {
        PackageStore pkg_store;
        pkg_store.push_back(pkg);
        _client_pkg_store.insert(std::make_pair(socket_list_id, pkg_store));
        client_store = _client_pkg_store.find(socket_list_id);
    } else {
        client_store->second.push_back(pkg);
    }
    _package_cache_size += byte_to_mb(pkg);
    _condition_empty_package.notify_one();
}

IPCPackage* SocketServer::pop_front_package(unsigned int socket_list_id) {
    boost::mutex::scoped_lock locker(_mutex_package);
    auto client_store = _client_pkg_store.find(socket_list_id);
    if (client_store == _client_pkg_store.end()) {
        return nullptr;
    } 
    if (client_store->second.size() == 0) {
        _client_pkg_store.erase(client_store);//clean empty sotre
        return nullptr;
    }
    IPCPackage* pkg = client_store->second.front();
    client_store->second.pop_front();

    _package_cache_size -= byte_to_mb(pkg);
    _condition_full_package.notify_one();
    return pkg;
}

void SocketServer::clear_package(unsigned int socket_list_id) {
    MI_UTIL_LOG(MI_INFO) << "try clear package.";
    PackageStore old_package_store;
    {
        boost::mutex::scoped_lock locker(_mutex_package);
        auto client_store = _client_pkg_store.find(socket_list_id);
        if (client_store != _client_pkg_store.end()) {
            MI_UTIL_LOG(MI_DEBUG) << "package store size: " << client_store->second.size();
            old_package_store = client_store->second;
            _client_pkg_store.erase(client_store);
        }   
    }

    MI_UTIL_LOG(MI_DEBUG) << "package store size tmp: " << old_package_store.size();
    while (!old_package_store.empty()) {
        IPCPackage* pkg = old_package_store.front();
        if (nullptr != pkg) {
            delete pkg;
            pkg = nullptr; 
        }
        old_package_store.pop_front();
    }
    MI_UTIL_LOG(MI_INFO) << "clear package success.";
}

void SocketServer::try_pop_front_package() {
    boost::mutex::scoped_lock locker(_mutex_package);
    while(_client_pkg_store.empty()) {
        _condition_empty_package.wait(_mutex_package);
    }
}

ServerStatus SocketServer::get_current_status() {
    ServerStatus status;
    {
        boost::mutex::scoped_lock locker(_mutex_package);
        for (auto it = _client_pkg_store.begin(); it != _client_pkg_store.end(); ++it) {
            status.client_packages.insert(std::make_pair(it->first,it->second.size()));          
        }
        const std::map<unsigned int, SocketList::SocketInfo> client_infos = _client_sockets->get_socket_infos();
        for (auto it = client_infos.begin(); it != client_infos.end(); ++it) {
            if (_socket_type == INET) {
                std::stringstream ss;
                ss << it->second.host << "::" << it->second.port;
                status.client_hosts.insert(std::make_pair(it->first,ss.str()));
            } else {
                status.client_hosts.insert(std::make_pair(it->first,it->second.host));
            }
        }
        status.cur_client = _client_sockets->get_socket_count();
        status.package_cache_size = _package_cache_size;
    }
    status.package_cache_capcity = _package_cache_capcity;
    if (_socket_type == UNIX) {
        status.socket_type = "UNIX";
        status.host = _path;
    } else if(_socket_type == INET) {
        status.socket_type = "INET";
        status.host = "localhost::" + _server_port;
    }
    return status;
}

MED_IMG_END_NAMESPACE