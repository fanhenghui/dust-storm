#ifndef TEST_COMMON_H
#define TEST_COMMON_H
#include <string>

struct Msg {
    int tag; // 0 string 1 data_byte -1 shut down tag
    int len;
    char* buffer;
    Msg() : tag(0), len(0), buffer(nullptr) {}
    ~Msg() {
        if (buffer) {
            delete[] buffer;
            buffer = nullptr;
        }
    }
};

#endif