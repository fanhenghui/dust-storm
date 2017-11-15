#ifndef MEDIMGUTIL_MI_SOCKET_LIST_H
#define MEDIMGUTIL_MI_SOCKET_LIST_H

#ifndef WIN32

#include "util/mi_util_export.h"
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/select.h>

#include <set>

#include "util/mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

class SocketList {
public:
    

    SocketList() {}

    ~SocketList() {}

    int get_socket_count() {
        boost::mutex::scoped_lock locker(_mutex);
        return static_cast<int>(_sockets.size());
    }

    int check_socket(int socket) {
        boost::mutex::scoped_lock locker(_mutex);
        return _sockets.find(socket) != _sockets.end() ? 0 : -1;
    }

    int insert_socket(int socket, const std::string addr, int port = 0) {
        boost::mutex::scoped_lock locker(_mutex);
        auto it = _sockets.find(socket);
        if (it != _sockets.end()) {
            MI_UTIL_LOG(MI_ERROR) << "insert the existed socket: " << socket;
            return -1;
        } else {
            _sockets.insert(socket);
            SocketAddr socket_addr;
            socket_addr.addr = addr;
            socket_addr.port = port;
            _sockets_addr[socket] = socket_addr;
            return 0;
        }
    }

    int remove_socket(int socket) {
        boost::mutex::scoped_lock locker(_mutex);
        MI_UTIL_LOG(MI_INFO) << "try remove socket: " << socket;
        auto it_addr = _sockets_addr.find(socket);
        if (it_addr != _sockets_addr.end()) {
            _sockets_addr.erase(it_addr);
        }

        auto it = _sockets.find(socket);
        if (it != _sockets.end()) {
            _sockets.erase(it);
            MI_UTIL_LOG(MI_INFO) << "remove socket: " << socket << " success";
            return 0;
        } else {
            MI_UTIL_LOG(MI_ERROR) << "remove the non-existed socket: " << socket;
            return -1;
        }
        
    }

    const std::set<int> get_sockets() const {
        return _sockets;
    }

    int get_socket_addr(int socket, std::string& addr, int& port) {
        auto it = _sockets_addr.find(socket);
        if (it != _sockets_addr.end()) {
            addr = it->second.addr;
            port = it->second.port;
            return 0;
        } else {
            return -1;
        }
    }

    int make_fd(fd_set* fds) {
        boost::mutex::scoped_lock locker(_mutex);
        FD_ZERO(fds);
        int max_fd = -1;
        for (auto it = _sockets.begin(); it != _sockets.end(); ++it) {
            max_fd = (*it) > max_fd ? (*it) : max_fd;
            FD_SET((*it), fds);
        }
        return max_fd;
    }

private:
    std::set<int> _sockets;
    struct SocketAddr {
        std::string addr;// ip for inet; path for unix
        int port;// port for inet
    };
    std::map<int, SocketAddr> _sockets_addr;
    boost::mutex _mutex;
private:
    DISALLOW_COPY_AND_ASSIGN(SocketList);
};

MED_IMG_END_NAMESPACE

#endif

#endif