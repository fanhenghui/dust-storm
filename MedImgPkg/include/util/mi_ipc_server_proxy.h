#ifndef MEDIMGUTIL_MI_IPC_SERVER_PROXY_H
#define MEDIMGUTIL_MI_IPC_SERVER_PROXY_H

#include "util/mi_util_export.h"
#include <memory>

#include "boost/noncopyable.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/thread.hpp"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class SocketServer;
class IPCServerProxy : public boost::noncopyable {
public:
    IPCServerProxy(SocketType type = UNIX);
    ~IPCServerProxy();

    void run();
    void send();
    void stop();
    void register_command_handler(unsigned int cmd_id,
                                  std::shared_ptr<ICommandHandler> handler);
    void unregister_command_handler(unsigned int cmd_id);
    void async_send_message(const IPCDataHeader& header, char* buffer);
    int handle_command(const IPCDataHeader& header, char* buffer);

protected:
private:
    std::unique_ptr<SocketServer> _server;
    std::map<unsigned int, std::shared_ptr<ICommandHandler>> _handlers;

    boost::mutex _mutex;
    boost::mutex _mutex_send_data;
private:
    DISALLOW_COPY_AND_ASSIGN(IPCServerProxy);
};

MED_IMG_END_NAMESPACE

#endif