
#ifndef MEDIMGUTIL_MI_SOCKET_CLIENT_H
#define MEDIMGUTIL_MI_SOCKET_CLIENT_H

#ifndef WIN32

#include "util/mi_util_export.h"

#include <string>
#include "boost/noncopyable.hpp"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class SocketClient {
public:
    SocketClient();
    ~SocketClient();

    void set_path(const std::string& path);
    std::string get_path() const;

    void register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler);
    void send_data(const IPCDataHeader& dataheader , char* buffer);

    void run();

protected:
private:
    std::string _path;
    std::shared_ptr<IPCDataRecvHandler> _handler;
    int _fd_server;

    static const int _reconnect_times = 100;

    pid_t _local_pid;
    pid_t _server_pid;

private:
    DISALLOW_COPY_AND_ASSIGN(SocketClient);
};

MED_IMG_END_NAMESPACE

#endif

#endif