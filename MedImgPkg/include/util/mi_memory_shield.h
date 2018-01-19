#ifndef MEDIMGUTIL_MI_MEMORY_SHIELD_H
#define MEDIMGUTIL_MI_MEMORY_SHIELD_H

#include "util/mi_util_export.h"

MED_IMG_BEGIN_NAMESPACE

class MemShield {
public:
    explicit MemShield(void* buffer):_buffer((char*)buffer) {};
    ~MemShield() {
        if(_buffer) {
            delete [] _buffer;
            _buffer = nullptr;
        }
    }

private:
    char* _buffer;
    DISALLOW_COPY_AND_ASSIGN(MemShield);
};

template<class T>
class StructShield {
public:
    explicit StructShield(T* struct_raw_pointer):_pointer(struct_raw_pointer) {};
    ~StructShield() {
        if(_pointer) {
            delete _pointer;
            _pointer = nullptr;
        }
    }

private:
    T* _pointer;
    DISALLOW_COPY_AND_ASSIGN(StructShield);
};

MED_IMG_END_NAMESPACE

#endif