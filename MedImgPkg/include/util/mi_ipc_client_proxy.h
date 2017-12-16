#ifndef MEDIMGUTIL_MI_IPC_CLIENT_PROXY_H
#define MEDIMGUTIL_MI_IPC_CLIENT_PROXY_H

#include "util/mi_util_export.h"
#include <memory>

#include "boost/thread/mutex.hpp"
#include "boost/thread/thread.hpp"

#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class SocketClient;
class IPCClientProxy {
public:
    explicit IPCClientProxy(SocketType type = UNIX);
    ~IPCClientProxy();

    void set_path(const std::string& path);
    void set_server_address(const std::string& ipv4, const std::string& port);

    void run();
    void stop();

    void register_command_handler(unsigned int cmd_id, std::shared_ptr<ICommandHandler> handler);
    void register_on_connection_event(std::shared_ptr<IEvent> ev);
    void unregister_command_handler(unsigned int cmd_id);
    int handle_command(const IPCDataHeader& header, char* buffer);

    int sync_send_data(const IPCDataHeader& header, char* buffer);
    int sync_send_data(IPCPackage* package);
    int sync_send_data(const std::vector<IPCPackage*>& packages);
    int sync_post(const std::vector<IPCPackage*>& packages);

protected:
private:
    std::unique_ptr<SocketClient> _client;
    std::map<unsigned int, std::shared_ptr<ICommandHandler>> _handlers;

    boost::mutex _mutex;
    boost::mutex _mutex_send_data;
private:
    DISALLOW_COPY_AND_ASSIGN(IPCClientProxy);
};

MED_IMG_END_NAMESPACE

#endif