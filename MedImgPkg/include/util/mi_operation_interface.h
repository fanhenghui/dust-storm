#ifndef MEDIMG_UTIL_MI_OPERATION_INTERFACE_H
#define MEDIMG_UTIL_MI_OPERATION_INTERFACE_H

#include <memory>

#include "util/mi_ipc_common.h"
#include "util/mi_controller_interface.h"

MED_IMG_BEGIN_NAMESPACE

class IOperation {
public:
    IOperation() : _buffer(nullptr) {
        memset((char*)(&_header), 0, sizeof(_header));
    };

    virtual ~IOperation() {
        if (nullptr != _buffer) {
            delete [] _buffer;
            _buffer = nullptr;
        }
    };

    void set_data(const IPCDataHeader& data, char* buffer) {
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
            delete[] _buffer;
            _buffer = nullptr;
        }
    }

    virtual std::shared_ptr<IOperation> create() = 0;

protected:
    IPCDataHeader _header;
    char* _buffer;
    std::weak_ptr<IController> _controller;

private:
    DISALLOW_COPY_AND_ASSIGN(IOperation);
};

#define CREATE_EXTENDS_OP(arg) virtual std::shared_ptr<IOperation> create() { \
     std::shared_ptr<IOperation> op(new arg); \
     return op;};

MED_IMG_END_NAMESPACE
#endif