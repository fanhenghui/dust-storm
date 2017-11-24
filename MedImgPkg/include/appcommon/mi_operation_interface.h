#ifndef MED_IMG_APPCOMMON_MI_OPERATION_INTERFACE_H
#define MED_IMG_APPCOMMON_MI_OPERATION_INTERFACE_H

#include "appcommon/mi_app_common_export.h"
#include <string.h>

#include "util/mi_ipc_common.h"
#include "appcommon/mi_app_common_logger.h"
#include "appcommon/mi_app_common_controller_interface.h"

MED_IMG_BEGIN_NAMESPACE

struct OpDataHeader {
    unsigned int cell_id;   // cell ID
    unsigned int op_id;     // operation ID
    unsigned int data_len;  // data length
    unsigned int receiver;  // receiver pid or socket id ...
    unsigned int end_tag;   // sequenced message end tag:0 for in the process 1 for the last message
    unsigned int reserved;  // IPC header msg_info3

    OpDataHeader():cell_id(0),op_id(0),data_len(0),receiver(0),end_tag(0),reserved(0)
    {}
};

class IOperation {
public:
    IOperation() : _buffer(nullptr) {
        memset((char*)(&_header), 0, sizeof(_header));
    };

    virtual ~IOperation() {
        if (nullptr != _buffer) {
            //MI_APPCOMMON_LOG(MI_DEBUG) << "delete buffer.";
            delete [] _buffer;
            _buffer = nullptr;
        }
    };

    void set_data(const OpDataHeader& data, char* buffer) {
        _header = data;
        _buffer = buffer;
    };

    void set_controller(std::shared_ptr<IController> controller) {
        _controller = controller;
    }

    template<class ControllerType>
    std::shared_ptr<ControllerType> get_controller() {
        std::shared_ptr<IController> i_controller = _controller.lock();
        if (nullptr == i_controller) {
            return nullptr;
        } else {
            return std::dynamic_pointer_cast<ControllerType>(i_controller);
        }
    }

    virtual int execute() {
        return 0;
    }

    void reset() {
        if (nullptr != _buffer) {
            //MI_APPCOMMON_LOG(MI_DEBUG) << "delete buffer.";
            delete[] _buffer;
            _buffer = nullptr;
        }
    }

    virtual std::shared_ptr<IOperation> create() = 0;

protected:
    OpDataHeader _header;
    char* _buffer;
    std::weak_ptr<IController> _controller;

private:
    DISALLOW_COPY_AND_ASSIGN(IOperation);
};

/* #define CREATE_MY_OP(arg) virtual std::shared_ptr<IOperation> create() { \
     std::shared_ptr<IOperation> op(new #arg); \
     return op;}
*/

MED_IMG_END_NAMESPACE
#endif