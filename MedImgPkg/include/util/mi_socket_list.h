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
#include <time.h>

#include <set>

#include "util/mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

class SocketList {
public:
    struct SocketInfo {
        int socket;
        unsigned int time;
        std::string host;// ip for inet; path for unix
        int port;// port for inet
    };

    SocketList():_cur_id(0) {}

    ~SocketList() {}

    int get_socket_count() {
        boost::mutex::scoped_lock locker(_mutex);
        return static_cast<int>(_socket_infos.size());
    }

    int check_socket(unsigned int id) {
        boost::mutex::scoped_lock locker(_mutex);
        return _socket_infos.find(id) != _socket_infos.end() ? 0 : -1;
    }

    int insert_socket(int socket, const std::string host, int port = 0) {
        boost::mutex::scoped_lock locker(_mutex);
        const unsigned int id = acquire_id_i();
        const unsigned int cur_time = (unsigned int)(time(NULL));
        auto it = _sockets.find(socket);
        if (it != _sockets.end()) {
            MI_UTIL_LOG(MI_ERROR) << "insert the existed socket: " << socket;
            return -1;
        } else {
            _sockets.insert(socket);
            SocketInfo socket_info;
            socket_info.host = host;
            socket_info.port = port;
            socket_info.socket = socket;
            socket_info.time = cur_time;
            _socket_infos[id] = socket_info;
            return 0;
        }
    }

    int remove_socket(unsigned int id) {
        boost::mutex::scoped_lock locker(_mutex);
        MI_UTIL_LOG(MI_INFO) << "try remove socket id: " << id;
        auto it_info = _socket_infos.find(id);
        if (it_info != _socket_infos.end()) {
            const int socket = it_info->second.socket;
            auto it_socket = _sockets.find(socket);
            if (it_socket != _sockets.end()) {
                _sockets.erase(it_socket);
            }
            _socket_infos.erase(it_info);
        }
        return 0;
    }

    const std::map<unsigned int, SocketList::SocketInfo>& get_socket_infos() const {
        boost::mutex::scoped_lock locker(_mutex);
        return _socket_infos;
    }

    int get_socket_info(unsigned int id , int& socket, unsigned int& time, std::string& host, int& port) {
        boost::mutex::scoped_lock locker(_mutex);
        auto it = _socket_infos.find(socket);
        if (it != _socket_infos.end()) {
            socket = it->second.socket;
            time = it->second.time;
            host = it->second.host;
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

    unsigned int reset_list() {
        boost::mutex::scoped_lock locker(_mutex);
        _cur_id = 0;
    }

private:
    unsigned int acquire_id_i() {
        return _cur_id++;
    }

private:
    std::set<int> _sockets;//for check
    std::map<unsigned int, SocketInfo> _socket_infos;
    unsigned int _cur_id;
    mutable boost::mutex _mutex;
private:
    DISALLOW_COPY_AND_ASSIGN(SocketList);
};

MED_IMG_END_NAMESPACE

#endif

#endif