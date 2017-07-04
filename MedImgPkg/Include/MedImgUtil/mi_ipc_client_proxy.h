#ifndef MED_IMG_IPC_CLIENT_PROXY_H
#define MED_IMG_IPC_CLIENT_PROXY_H

#include "MedImgUtil/mi_util_export.h"
#include <memory>

#include "boost/thread/mutex.hpp"
#include "boost/thread/thread.hpp"
#include "MedImgUtil/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE


class SocketClient;
class IPCClientProxy
{
public:
    IPCClientProxy();
    ~IPCClientProxy();

    void set_path(const std::string& path);
    void run();
    void register_command_handler(unsigned int cmd_id , std::shared_ptr<ICommandHandler> handler);
    void unregister_command_handler(unsigned int cmd_id);
    void async_send_message(const IPCDataHeader& header , void* buffer);
    int handle_command(const IPCDataHeader& header , void* buffer);

protected:
    
private:
    std::unique_ptr<SocketClient> _client;
    std::map<unsigned int , std::shared_ptr<ICommandHandler>> _handlers;

    boost::mutex _mutex;
};

MED_IMG_END_NAMESPACE


#endif