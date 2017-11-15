
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
    SocketClient(SocketType type = UNIX);
    ~SocketClient();

    //for AF_UNIX
    void set_path(const std::string& path);
    std::string get_path() const;

    //for AF_INET
    void set_server_address(const std::string& ipv4, const std::string& port);
    void get_server_address(std::string& ipv4, std::string& port) const;

    void register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler);
    void sync_send_data(const IPCDataHeader& dataheader, char* buffer);

    void run();
    void stop();

    int sync_post(const IPCDataHeader& post_header, char* post_data, IPCDataHeader& result_header, char*& result_data);
    int sync_post(const std::vector<IPCPackage*>& packages);

private:
    void connect_i();
private:
    SocketType _socket_type;
    
    //address for AF_UNIX
    std::string _path;
    //address for AF_INET
    std::string _server_ip;
    std::string _server_port;

    std::shared_ptr<IPCDataRecvHandler> _handler;
    int _fd_server;

    static const int _reconnect_times = 100;
    bool _alive;
private:
    DISALLOW_COPY_AND_ASSIGN(SocketClient);
};

MED_IMG_END_NAMESPACE

#endif

#endif