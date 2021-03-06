#include "mi_aabb.h"

#ifdef WIN32

#else
#include <string.h>
#endif

MED_IMG_BEGIN_NAMESPACE

AABB::AABB() {}

AABB::~AABB() {}

bool AABB::operator==(const AABB& aabb) const {
    return _min == aabb._min && _max == aabb._max;
}

bool AABB::operator!=(const AABB& aabb) const {
    return _min != aabb._min || _max != aabb._max;
}

AABBUI::AABBUI() {
    memset(_min, 0, sizeof(_min));
    memset(_max, 0, sizeof(_max));
}

AABBUI::AABBUI(const unsigned int (&min0)[3], const unsigned int (&max0)[3]) {
    memcpy(_min, min0, sizeof(_min));
    memcpy(_max, max0, sizeof(_max));
}

AABBUI::~AABBUI() {}

bool AABBUI::operator==(const AABBUI& aabb) const {
    return (_min[0] == aabb._min[0] && _min[1] == aabb._min[1] &&
            _min[2] == aabb._min[2] && _max[0] == aabb._max[0] &&
            _max[1] == aabb._max[1] && _max[2] == aabb._max[2]);
}

bool AABBUI::operator!=(const AABBUI& aabb) const {
    return (_min[0] != aabb._min[0] || _min[1] != aabb._min[1] ||
            _min[2] != aabb._min[2] || _max[0] != aabb._max[0] ||
            _max[1] != aabb._max[1] || _max[2] != aabb._max[2]);
}

int AABBUI::volume() const {
    return (_max[0] - _min[0])*(_max[1] - _min[1])*(_max[2] - _min[2]);
}


AABBI::AABBI() {
    memset(_min, 0, sizeof(_min));
    memset(_max, 0, sizeof(_max));
}

AABBI::AABBI(const int (&min0)[3], const int (&max0)[3]) {
    memcpy(_min, min0, sizeof(_min));
    memcpy(_max, max0, sizeof(_max));
}

AABBI::~AABBI() {}

bool AABBI::operator==(const AABBI& aabb) const {
    return (_min[0] == aabb._min[0] && _min[1] == aabb._min[1] &&
            _min[2] == aabb._min[2] && _max[0] == aabb._max[0] &&
            _max[1] == aabb._max[1] && _max[2] == aabb._max[2]);
}

bool AABBI::operator!=(const AABBI& aabb) const {
    return (_min[0] != aabb._min[0] || _min[1] != aabb._min[1] ||
            _min[2] != aabb._min[2] || _max[0] != aabb._max[0] ||
            _max[1] != aabb._max[1] || _max[2] != aabb._max[2]);
}

int AABBI::volume() const {
    return (_max[0] - _min[0])*(_max[1] - _min[1])*(_max[2] - _min[2]);
}


MED_IMG_END_NAMESPACE