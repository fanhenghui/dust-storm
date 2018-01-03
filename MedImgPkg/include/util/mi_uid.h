#ifndef MED_IMG_UTIL_MI_UID_H
#define MED_IMG_UTIL_MI_UID_H

#include "util/mi_util_export.h"
#include <boost/thread/mutex.hpp>

MED_IMG_BEGIN_NAMESPACE

typedef long long UIDType;

class UIDGenerator {
public:
    UIDGenerator() : _base(0) {}

    ~UIDGenerator() {
        _base = 0;
    }

    void reset() {
        boost::unique_lock<boost::mutex> locker(_mutex);
        _base = 0;
    }

    UIDType tick() {
        boost::unique_lock<boost::mutex> locker(_mutex);
        return _base++;
    }

private:
    UIDType _base;
    boost::mutex _mutex;
};

MED_IMG_END_NAMESPACE
#endif