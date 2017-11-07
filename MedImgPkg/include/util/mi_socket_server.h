#ifndef MEDIMGUTIL_MI_SOCKET_SERVER_H
#define MEDIMGUTIL_MI_SOCKET_SERVER_H

#include "util/mi_util_export.h"

#include <string>
#include <deque>
#include <map>
#include "boost/thread/thread.hpp"
#include "boost/thread/mutex.hpp"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class SocketServer {
public:
    SocketServer(SocketType type = UNIX);
    ~SocketServer();

    void set_max_client(int max_clients);

    //for AF_UNIX
    void set_path(const std::string& path);
    std::string get_path() const;

    //for AF_INET (Just Port)
    void set_server_address(const std::string& port);
    void get_server_address(std::string& port) const;

    void register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler);
    void send_data(const IPCDataHeader& dataheader , char* buffer);

    void run();
    void stop();


protected:
private:
    //address for AF_UNIX
    std::string _path;
    //address for AF_INET
    std::string _server_port;

    //std::vector<int> _socket_client;

    std::shared_ptr<IPCDataRecvHandler> _handler;
    int _fd_server;
    int _max_clients;

    SocketType _socket_type;
    bool _alive;

    struct Package {
        IPCDataHeader header;
        char* buffer;

        Package(): buffer(nullptr) {};
        Package(const IPCDataHeader&header_, char* buffer_):
            header(header_),buffer(buffer_) {}
        ~Package() {
            if (nullptr !=  buffer) {
                delete [] buffer;
            }
        }
    };
    typedef std::deque<Package*> PackageStore;
    std::map<int , PackageStore> _client_pkg_store;

private:
    DISALLOW_COPY_AND_ASSIGN(SocketServer);
};

MED_IMG_END_NAMESPACE

#endif