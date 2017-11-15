#ifndef WIN32
#include "mi_ipc_client_proxy.h"
#include "mi_socket_client.h"
#include "mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

class IPCDataRecvHandlerExt : public IPCDataRecvHandler {
public:
    IPCDataRecvHandlerExt(IPCClientProxy* proxy) : _proxy(proxy) {
    };

    virtual ~IPCDataRecvHandlerExt() {
    };

    virtual int handle(const IPCDataHeader& header , char* buffer) {
        if (_proxy) {
            return _proxy->handle_command(header , buffer);
        } else {
            MI_UTIL_LOG(MI_ERROR) << "IPC client proxy is null when handle IPC data received.";
            return 0;
        }
    }
protected:
private:
    IPCClientProxy* _proxy;
};

IPCClientProxy::IPCClientProxy(SocketType type): _client(new SocketClient(type)) {
    std::shared_ptr<IPCDataRecvHandlerExt> recv_handler(new IPCDataRecvHandlerExt(this));
    _client->register_revc_handler(recv_handler);
}

IPCClientProxy::~IPCClientProxy() {

}

void IPCClientProxy::set_path(const std::string& path) {
    _client->set_path(path);
}

void IPCClientProxy::set_server_address(const std::string& ipv4, const std::string& port) {
    _client->set_server_address(ipv4, port);
}

void IPCClientProxy::run() {
    _client->run();
}

void IPCClientProxy::register_command_handler(unsigned int cmd_id ,
        std::shared_ptr<ICommandHandler> handler) {
    boost::mutex::scoped_lock locker(_mutex);
    _handlers[cmd_id] = handler;
}

void IPCClientProxy::unregister_command_handler(unsigned int cmd_id) {
    boost::mutex::scoped_lock locker(_mutex);

    auto it = _handlers.find(cmd_id);
    if (it != _handlers.end()) {
        _handlers.erase(it);
    }
}

void IPCClientProxy::sync_send_data(const IPCDataHeader& header , char* buffer) {
    boost::mutex::scoped_lock locker(_mutex_send_data);
    _client->sync_send_data(header , buffer);
}

int IPCClientProxy::handle_command(const IPCDataHeader& header , char* buffer) {
    boost::mutex::scoped_lock locker(_mutex);

    const unsigned int cmd_id = header.msg_id;
    auto it = _handlers.find(cmd_id);

    if (it != _handlers.end() && it->second) {
        return it->second->handle_command(header , buffer);
    } else {
        if (nullptr != buffer ) {
            delete [] buffer;
        }
        MI_UTIL_LOG(MI_WARNING) << "cant find handler to process ipc data. header detail : " << STREAM_IPCHEADER_INFO(header);
        return -1;
    }
}

int IPCClientProxy::sync_post(const std::vector<IPCPackage*>& packages) {
    boost::mutex::scoped_lock locker(_mutex_send_data);
    _client->sync_post(packages);
}

MED_IMG_END_NAMESPACE

#endif


