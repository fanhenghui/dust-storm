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
    explicit SocketServer(SocketType type = UNIX);
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
    int set_package_cache_capcity(float capcity_mb);//default 1024.0f*10.0f 10GB

    void run();
    void stop();
    int  recv();
    int  send();
    
    ServerStatus get_current_status();
private:
    IPCPackage* pop_front_package(unsigned int socket_list_id);
    void try_pop_front_package();
    void push_back_package(unsigned int socket_list_id, IPCPackage* pkg);
    void clear_package(unsigned int socket_list_id);

    inline float byte_to_mb(IPCPackage* pkg) {
        const static float val = 1.0f / 1024.0f / 1024.0f;
        return pkg == nullptr ? 0 : pkg->header.data_len * val;
    }
    inline bool package_cache_full(IPCPackage* pkg_to_be_push) {
        return (_package_cache_size + byte_to_mb(pkg_to_be_push)) >= _package_cache_capcity;
    } 
private:
    SocketType _socket_type;

    //address for AF_UNIX
    std::string _path;
    //address for AF_INET
    std::string _server_port;

    std::shared_ptr<IPCDataRecvHandler> _handler;
    int _fd_server;
    int _max_clients; // TODO add client limit
    
    bool _alive;

    //client socket fd
    std::shared_ptr<SocketList> _client_sockets;

    //package to be sending to client
    typedef std::deque<IPCPackage*> PackageStore;
    std::map<unsigned int , PackageStore> _client_pkg_store;

    boost::mutex _mutex_package;
    boost::condition _condition_empty_package;
    boost::condition _condition_full_package;
    float _package_cache_capcity;
    float _package_cache_size;
    
private:
    DISALLOW_COPY_AND_ASSIGN(SocketServer);
};

MED_IMG_END_NAMESPACE

#endif
#endif