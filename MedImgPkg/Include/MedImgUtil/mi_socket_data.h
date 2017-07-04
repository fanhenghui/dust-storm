#ifndef MED_IMG_SOCKET_DATA_H_
#define MED_IMG_SOCKET_DATA_H_

#include "MedImgUtil/mi_util_export.h"

MED_IMG_BEGIN_NAMESPACE

struct IPCDataHeader
{
    unsigned int _sender;//sender pid
    unsigned int _receiver;//receiver pid
    unsigned int _msg_id;//command ID
    unsigned int _data_type;//0 raw_data 1 protocol buffer
    unsigned int _big_end;//0 small end 1 big_end 
    unsigned int _data_len;//data length
};

class IPCDataRecvHandler
{
public:
    virtual ~IPCDataRecvHandler() {};
    void handle(const &IPCDataHeader header , void* buffer) = 0;
protected:
private:
};

MED_IMG_END_NAMESPACE

#endif