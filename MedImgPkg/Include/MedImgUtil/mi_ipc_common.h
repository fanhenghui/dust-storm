#ifndef MED_IMG_SOCKET_DATA_H_
#define MED_IMG_SOCKET_DATA_H_

#include "MedImgUtil/mi_util_export.h"

#include <memory>

MED_IMG_BEGIN_NAMESPACE

struct IPCDataHeader
{
    unsigned int _sender;//sender pid
    unsigned int _receiver;//receiver pid
    unsigned int _msg_id;//message ID : thus command ID
    unsigned int _msg_info0;//message info : thus cell ID
    unsigned int _msg_info1;//message info : thus operation ID
    unsigned int _data_type;//0 raw_data 1 protocol buffer
    unsigned int _big_end;//0 small end 1 big_end 
    unsigned int _data_len;//data length
};

class IPCDataRecvHandler
{
public:
    IPCDataRecvHandler() {};
    virtual ~IPCDataRecvHandler() {};
    virtual int handle(const IPCDataHeader& header , void* buffer) = 0;
protected:
private:
};

class ICommandHandler
{
public:
    ICommandHandler() {};
    virtual ~ICommandHandler() {};
    virtual int handle_command(const IPCDataHeader& datahaeder , void* buffer) = 0;
protected:
private:
};


MED_IMG_END_NAMESPACE

#endif