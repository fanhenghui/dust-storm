#include "mi_ipc_client_proxy.h"
#include "mi_socket_client.h"

MED_IMG_BEGIN_NAMESPACE

class IPCDataRecvHandlerExt : public IPCDataRecvHandler
{
public:
    IPCDataRecvHandlerExt(IPCClientProxy* proxy):_proxy(proxy) 
    {};

    virtual ~IPCDataRecvHandlerExt() 
    {};

    virtual int handle(const IPCDataHeader& header , void* buffer)
    {
        if(_proxy){
            return _proxy->handle_command(header , buffer);
        }
        else{
            //TODO return what number
            return 0;
        }
    }
protected:
private:
    IPCClientProxy* _proxy;
}; 

IPCClientProxy::IPCClientProxy():_client(new SocketClient())
{

}

IPCClientProxy::~IPCClientProxy()
{

}

void IPCClientProxy::set_path(const std::string& path)
{
    _client->set_path(path);
}

void IPCClientProxy::run()
{
    _client->run();
}

void IPCClientProxy::register_command_handler(unsigned int cmd_id , std::shared_ptr<ICommandHandler> handler)
{
    boost::mutex::scoped_lock locker(_mutex);
    _handlers[cmd_id] = handler;
}

void IPCClientProxy::unregister_command_handler(unsigned int cmd_id)
{
    boost::mutex::scoped_lock locker(_mutex);

    auto it = _handlers.find(cmd_id);
    if(it != _handlers.end()){
        _handlers.erase(it);
    }
}

void IPCClientProxy::async_send_message(const IPCDataHeader& header , void* buffer)
{
    _client->send_data(header , buffer);
}

int IPCClientProxy::handle_command(const IPCDataHeader& header , void* buffer)
{
    boost::mutex::scoped_lock locker(_mutex);

    const unsigned int cmd_id = header._msg_id;
    auto it = _handlers.find(cmd_id);
    if(it != _handlers.end() && it->second){
        return it->second->handle_command(header , buffer);
    }
    else{
        //TODO return what number
        return 0;
    }
}

MED_IMG_END_NAMESPACE


