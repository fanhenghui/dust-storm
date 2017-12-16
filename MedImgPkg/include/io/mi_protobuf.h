#ifndef MEDIMG_IO_MI_PROTOBUF_H
#define MEDIMG_IO_MI_PROTOBUF_H

#include "io/mi_io_export.h"
#include "io/mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

template<class MsgType>
int protobuf_decode(char* buffer, int size, MsgType& msg) {
    if (buffer==nullptr || size<=0) {
        return -1;
    }
    if (!msg.ParseFromArray(buffer, size)) {
        msg.Clear();
        return -1;
    }
    return 0;
}

template<class MsgType>
int protobuf_decode(MsgType& msg, char*& buffer, int& size) {
    if (nullptr != buffer) {
        return -1;
    }
    size = msg.Byte();
    if (size <= 0) {
        return -1;
    }
    buffer = new char[size];
    if (nullptr == buffer) {
        return -1;
    }
    if (!msg.SerializeToArray(buffer,size)) {
        size = 0;
        delete [] buffer;
        buffer = nullptr;
        return -1;
    } else {
        return 0;
    }
}

MED_IMG_END_NAMESPACE

#endif