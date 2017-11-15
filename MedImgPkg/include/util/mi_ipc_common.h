#ifndef MEDIMGUTIL_MI_IPC_COMMON_H
#define MEDIMGUTIL_MI_IPC_COMMON_H

#include "util/mi_util_export.h"

#include <memory>

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
    unsigned int data_type;//0 raw_data 1 protocol buffer
    unsigned int big_end;//0 small end 1 big_end
    unsigned int data_len;//data length

    IPCDataHeader():
        sender(0), receiver(0), msg_id(0), msg_info0(0),
        msg_info1(0), data_type(0), big_end(0), data_len(0) {
    }
};
#define STREAM_IPCHEADER_INFO(header) "sender: " << header.sender << \
    "; receiver: " << header.receiver << \
    "; cmd id: " << header.msg_id << \
    "; cell id: " << header.msg_info0 << \
    "; op id: " << header.msg_info1 << \
    "; data type: " << header.data_type << \
    "; big end: " << header.big_end << \
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
    virtual int handle_command(const IPCDataHeader& datahaeder , char* buffer) = 0;
protected:
private:
};


const int CLIENT_QUIT_ID = 911119;

MED_IMG_END_NAMESPACE

#endif