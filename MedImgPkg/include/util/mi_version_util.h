#ifndef MED_IMG_UTIL_MI_VERSION_UTIL_H
#define MED_IMG_UTIL_MI_VERSION_UTIL_H

#include "util/mi_util_export.h"
#include <string>
#include <ostream>

MED_IMG_BEGIN_NAMESPACE

class Version {
public:
    int _major;
    int _minor;
    int _revision;

public:
    Version() : _major(0),_minor(0),_revision(0) {}

    Version(int major_, int minor_, int revision_) : 
    _major(major_),_minor(minor_),_revision(revision_) {}
};

inline bool operator < (const Version& l, const Version& r) {
    return (l._major*1e8 + l._minor*1e4 + l._revision) - (r._major*1e8 + r._minor*1e4 + r._revision) < 0.0f; 
}

inline bool operator > (const Version& l, const Version& r) {
    return (l._major*1e8 + l._minor*1e4 + l._revision) - (r._major*1e8 + r._minor*1e4 + r._revision) > 0.0f; 
}

inline bool operator == (const Version& l, const Version& r) {
    return l._major == r._major && l._minor == r._minor && l._revision == r._revision;
}

inline std::ostream& operator << (std::ostream& s, const Version& v) {
    s << v._major << "." << v._minor << "." << v._revision;
    return s;
}

//format: major.minor.revsion
inline int make_version(const std::string& str, Version& v) {
    int p_count = 0;
    int str_seg[2] = {-1,-1};
    for (size_t i = 0; i< str.size(); ++i) {
        if (str[i] >= '0' && str[i] <='9') {
            continue;
        } else if (str[i] == '.') {
            if (p_count > 1) {
                return -1;
            }
            str_seg[p_count++] = i;
        } else {
            return -1;
        }
    }

    if (p_count != 2) {
        return -1;
    }
    if (str_seg[1] == (int)str.size() - 1) {
        return -1;
    }
    
    v._major = atoi(str.substr(0,str_seg[0]).c_str());
    v._minor = atoi(str.substr(str_seg[0]+1,str_seg[1]-str_seg[0]).c_str());
    v._revision = atoi(str.substr(str_seg[1]+1,str.size()-str_seg[1]-1).c_str());
    return 0;
}

MED_IMG_END_NAMESPACE

#endif
