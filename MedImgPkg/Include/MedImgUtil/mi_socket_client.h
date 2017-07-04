#ifndef WIN32

#ifndef MED_IMG_SOCKET_CLIENT_H_
#define MED_IMG_SOCKET_CLIENT_H_

#include "MedImgUtil/mi_util_export.h"

MED_IMG_BEGIN_NAMESPACE

class SocketClient
{
public:
    SocketClient();
    ~SocketClient();

    void set_path(const std::string& path);
    std::string get_path() const;

    void register_revc_handler(std::shared_ptr<IPCDataRecvHandler> handler);
    void send(const IPCDataHeader& data , void* buffer);

    void run();

protected:
private:
    std::string _path;
    std::shared_ptr<IPCDataRecvHandler> _handler;
    bool _accepted;
};

MED_IMG_END_NAMESPACE

#endif

#endif