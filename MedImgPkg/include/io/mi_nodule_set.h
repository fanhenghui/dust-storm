#ifndef MEDIMGIO_NODULE_SET_H
#define MEDIMGIO_NODULE_SET_H

#include "io/mi_voi.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class IO_Export NoduleSet {
public:
    NoduleSet() {}
    ~NoduleSet() {}

    void add_nodule(const VOISphere& v) {
        _nodule_set.push_back(v);
    }

    void set_nodule(const std::vector<VOISphere>& nset) {
        _nodule_set = nset;
    }
    
    void clear_nodule() {
        std::vector<VOISphere>().swap(_nodule_set);
    }

    const std::vector<VOISphere>& get_nodule_set() const {
        return _nodule_set;
    }

    void get_nodule_set(std::vector<VOISphere>& nset) const {
        nset = _nodule_set;
    }

private:
    std::vector<VOISphere> _nodule_set;
};

MED_IMG_END_NAMESPACE

#endif