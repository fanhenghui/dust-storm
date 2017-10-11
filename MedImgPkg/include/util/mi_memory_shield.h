#ifndef MEDIMGUTIL_MI_MEMORY_SHIELD_H
#define MEDIMGUTIL_MI_MEMORY_SHIELD_H

#include "util/mi_util_export.h"

MED_IMG_BEGIN_NAMESPACE

class MemShield {
public:
    MemShield(char* buffer):_buffer(buffer) {};
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

MED_IMG_END_NAMESPACE

#endif