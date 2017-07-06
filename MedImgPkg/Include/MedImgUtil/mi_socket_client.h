#ifndef WIN32

#ifndef MED_IMG_SOCKET_CLIENT_H_
#define MED_IMG_SOCKET_CLIENT_H_

#include "MedImgUtil/mi_util_export.h"

#include <string>

#include "MedImgUtil/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class SocketClient
{
public:
    SocketClient();
    ~SocketClient();

    void set_path(const std::string& path);
    std::string get_path() const;

    void register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler);
    void send_data(const IPCDataHeader& dataheader , void* buffer);

    void run();

protected:
private:
    std::string _path;
    std::shared_ptr<IPCDataRecvHandler> _handler;
    int _fd_server;

    static const int _reconnect_times = 100;

    pid_t _local_pid;
    pid_t _server_pid;
};

MED_IMG_END_NAMESPACE

#endif

#endif