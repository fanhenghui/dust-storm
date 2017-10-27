#ifndef MED_IMG_APPCOMMON_MI_OPERATION_INTERFACE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_INTERFACE_H

#include "appcommon/mi_app_common_export.h"
#include <string.h>

#include "util/mi_ipc_common.h"
#include "appcommon/mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

struct OpDataHeader {
    unsigned int _cell_id;   //
    unsigned int _op_id;     // operation id
    unsigned int _data_type; // 0 raw_data 1 protocol buffer
    unsigned int _big_end;   // 0 small end 1 big_end
    unsigned int _data_len;  // data length
};

class AppController;
class IOperation {
public:
    IOperation() : _buffer(nullptr) {
        memset((char*)(&_header), 0, sizeof(_header));
    };

    virtual ~IOperation() {
        if (nullptr != _buffer) {
            MI_APPCOMMON_LOG(MI_DEBUG) << "delete buffer.";
            delete [] _buffer;
            _buffer = nullptr;
        }
    };

    void set_data(const OpDataHeader& data, char* buffer) {
        _header = data;
        _buffer = buffer;
    };

    void set_controller(std::shared_ptr<AppController> controller) {
        _controller = controller;
    }

    virtual int execute() {
        return 0;
    }

    void reset() {
        if (nullptr != _buffer) {
            MI_APPCOMMON_LOG(MI_DEBUG) << "delete buffer.";
            delete[] _buffer;
            _buffer = nullptr;
        }
    }

    virtual std::shared_ptr<IOperation> create() = 0;

protected:
    OpDataHeader _header;
    char* _buffer;
    std::weak_ptr<AppController> _controller;

private:
    DISALLOW_COPY_AND_ASSIGN(IOperation);
};

// #define CREATE_MY_OP(arg) virtual std::shared_ptr<IOperation> create() { \
//     std::shared_ptr<IOperation> op(new #arg); \
//     return op;}

MED_IMG_END_NAMESPACE
#endif