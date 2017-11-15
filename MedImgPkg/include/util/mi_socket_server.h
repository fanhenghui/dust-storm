#ifndef MEDIMGUTIL_MI_SOCKET_SERVER_H
#define MEDIMGUTIL_MI_SOCKET_SERVER_H

#ifndef WIN32

#include "util/mi_util_export.h"

#include <string>
#include <deque>
#include <map>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class SocketList;
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
    int async_send_data(IPCPackage* package);//return 0:success; -1:client disconnect 

    void run();
    void stop();
    int  send();
private:
    IPCPackage* pop_front_package_i(unsigned int socket_list_id);
    void try_pop_front_package_i();
    void push_back_package_i(unsigned int socket_list_id, IPCPackage* pkg);
    void clear_package_i(unsigned int socket_list_id);
private:
    //address for AF_UNIX
    std::string _path;
    //address for AF_INET
    std::string _server_port;

    //std::vector<int> _socket_client;

    std::shared_ptr<IPCDataRecvHandler> _handler;
    int _fd_server;
    int _max_clients; // TODO add client limit

    SocketType _socket_type;
    bool _alive;

    //client socket fd
    std::shared_ptr<SocketList> _client_sockets;

    //package to be sending to client
    typedef std::deque<IPCPackage*> PackageStore;
    std::map<unsigned int , PackageStore> _client_pkg_store;

    boost::mutex _mutex_package;
    boost::condition _condition_empty_package;
    
private:
    DISALLOW_COPY_AND_ASSIGN(SocketServer);
};

MED_IMG_END_NAMESPACE

#endif
#endif