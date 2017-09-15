#ifndef MEDIMGUTIL_MI_IPC_COMMON_H
#define MEDIMGUTIL_MI_IPC_COMMON_H

#include "util/mi_util_export.h"

#include <memory>

MED_IMG_BEGIN_NAMESPACE

struct IPCDataHeader { //32 byte
    unsigned int _sender;//sender pid
    unsigned int _receiver;//receiver pid
    unsigned int _msg_id;//message ID : thus command ID
    unsigned int _msg_info0;//message info : thus cell ID
    unsigned int _msg_info1;//message info : thus operation ID
    unsigned int _data_type;//0 raw_data 1 protocol buffer
    unsigned int _big_end;//0 small end 1 big_end
    unsigned int _data_len;//data length

    IPCDataHeader():
        _sender(0), _receiver(0), _msg_id(0), _msg_info0(0),
        _msg_info1(0), _data_type(0), _big_end(0), _data_len(0) {
    }
};
#define STREAM_IPCHEADER_INFO(header) "sender: " << header._sender << \
    "; receiver: " << header._receiver << \
    "; cmd id: " << header._msg_id << \
    "; cell id: " << header._msg_info0 << \
    "; op id: " << header._msg_info1 << \
    "; data type: " << header._data_type << \
    "; big end: " << header._big_end << \
    "; data len: " << header._data_len


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