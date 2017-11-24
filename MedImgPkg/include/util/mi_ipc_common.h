#ifndef MEDIMGUTIL_MI_IPC_COMMON_H
#define MEDIMGUTIL_MI_IPC_COMMON_H

#include "util/mi_util_export.h"

#include <memory>
#include <map>

MED_IMG_BEGIN_NAMESPACE

enum SocketType {
    INET,
    UNIX,//just for linux
};

struct IPCDataHeader { //32 byte
    unsigned int sender;//sender pid or socket id ... 
    unsigned int receiver;//receiver pid or socket id ...
    unsigned int msg_id;//message ID : thus command ID
    unsigned int msg_info0;//message info : client cell ID, client socket time 
    unsigned int msg_info1;//message info : client operation ID
    unsigned int msg_info2;//message info : reserved sequenced msg end tag(EG: send n slice dicom series. 0~n-2:0 n-1:1)
    unsigned int msg_info3;//message info : reserved
    unsigned int data_len;//data length

    IPCDataHeader():
        sender(0), receiver(0), msg_id(0), msg_info0(0),
        msg_info1(0), msg_info2(0), msg_info3(0), data_len(0) {
    }
};
#define STREAM_IPCHEADER_INFO(header) "sender: " << header.sender << \
    "; receiver: " << header.receiver << \
    "; cmd id: " << header.msg_id << \
    "; msg info0(cell id): " << header.msg_info0 << \
    "; msg info1(op id): " << header.msg_info1 << \
    "; msg info2(end tag): " << header.msg_info2 << \
    "; msg info3(reserved): " << header.msg_info3 << \
    "; data len: " << header.data_len

struct IPCPackage {
    IPCDataHeader header;
    char* buffer;
    bool ref;
    IPCPackage(const IPCDataHeader& header_, char* buffer_, bool ref_=false):
        header(header_),buffer(buffer_),ref(ref_) {}
    IPCPackage(const IPCDataHeader& header_):
        header(header_),buffer(nullptr),ref(false) {}
    ~IPCPackage() {
        if (!ref) {
            if (nullptr != buffer) {
                delete [] buffer;
                buffer = nullptr;
            }
        }
    }
};

struct ServerStatus {
    std::string socket_type;//UNIX/INET
    std::string host;
    int cur_client;
    int package_cache_capcity;
    int package_cache_size;
    std::map<unsigned int, size_t> client_packages;//client ID, package to be send
    std::map<unsigned int, std::string> client_hosts;
};

class IPCDataRecvHandler {
public:
    IPCDataRecvHandler() {};
    virtual ~IPCDataRecvHandler() {};
    virtual int handle(const IPCDataHeader& header , char* buffer) = 0;
protected:
private:
};

class ICommandHandler {
public:
    ICommandHandler() {};
    virtual ~ICommandHandler() {};
    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer) = 0;
protected:
private:
};


const int CLIENT_QUIT_ID = 911119;

MED_IMG_END_NAMESPACE

#endif