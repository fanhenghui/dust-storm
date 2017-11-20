#ifndef WIN32
#include "mi_ipc_server_proxy.h"
#include "mi_socket_server.h"
#include "mi_util_logger.h"

MED_IMG_BEGIN_NAMESPACE

class IPCDataRecvHandlerExt : public IPCDataRecvHandler {
public:
    IPCDataRecvHandlerExt(IPCServerProxy* proxy) : _proxy(proxy) {
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
    IPCServerProxy* _proxy;
};

IPCServerProxy::IPCServerProxy(SocketType type): _server(new SocketServer(type)) {
    std::shared_ptr<IPCDataRecvHandlerExt> recv_handler(new IPCDataRecvHandlerExt(this));
    _server->register_revc_handler(recv_handler);
}

IPCServerProxy::~IPCServerProxy() {

}

void IPCServerProxy::run() {
    _server->run();
}

void IPCServerProxy::send() {
    _server->send();
}

void IPCServerProxy::recv() {
    _server->recv();
}

void IPCServerProxy::stop() {
    _server->stop();
}

void IPCServerProxy::register_command_handler(unsigned int cmd_id ,
        std::shared_ptr<ICommandHandler> handler) {
    boost::mutex::scoped_lock locker(_mutex);
    _handlers[cmd_id] = handler;
}

void IPCServerProxy::unregister_command_handler(unsigned int cmd_id) {
    boost::mutex::scoped_lock locker(_mutex);

    auto it = _handlers.find(cmd_id);
    if (it != _handlers.end()) {
        _handlers.erase(it);
    }
}

int IPCServerProxy::async_send_data(IPCPackage* package) {
    boost::mutex::scoped_lock locker(_mutex_send_data);
    return _server->async_send_data(package);
}

int IPCServerProxy::handle_command(const IPCDataHeader& header , char* buffer) {
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

ServerStatus IPCServerProxy::get_current_status() {
    return _server->get_current_status();
}

MED_IMG_END_NAMESPACE

#endif


