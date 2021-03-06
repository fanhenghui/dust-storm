
#ifndef MEDIMGUTIL_MI_SOCKET_CLIENT_H
#define MEDIMGUTIL_MI_SOCKET_CLIENT_H

#ifndef WIN32

#include "util/mi_util_export.h"

#include <string>
#include <vector>

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class SocketClient {
public:
    explicit SocketClient(SocketType type = UNIX);
    ~SocketClient();

    //for AF_UNIX
    void set_path(const std::string& path);
    std::string get_path() const;

    //for AF_INET
    void set_server_address(const std::string& ipv4, const std::string& port);
    void get_server_address(std::string& ipv4, std::string& port) const;

    void register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler);
    int sync_send_data(const IPCDataHeader& dataheader, char* buffer);
    int sync_send_data(IPCPackage* package);
    int sync_send_data(const std::vector<IPCPackage*>& packages);

    void run();
    void stop();

    int sync_post(const std::vector<IPCPackage*>& packages);

    void on_connect(std::shared_ptr<IEvent> ev);

private:
    void connect();
private:
    SocketType _socket_type;
    
    //address for AF_UNIX
    std::string _path;
    //address for AF_INET
    std::string _server_ip;
    std::string _server_port;
    //info for debug/log
    std::string _server_info;

    int _fd_server;

    std::shared_ptr<IPCDataRecvHandler> _handler;
    std::shared_ptr<IEvent> _on_connect_event;
    
    static const int _reconnect_times = 100;
private:
    DISALLOW_COPY_AND_ASSIGN(SocketClient);
};

MED_IMG_END_NAMESPACE

#endif

#endif