#ifndef MEDIMGIO_NODULE_SET_H
#define MEDIMGIO_NODULE_SET_H

#include "io/mi_voi.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class IO_Export NoduleSet {
public:
    NoduleSet();
    ~NoduleSet();

    void add_nodule(const VOISphere& v);
    void set_nodule(const std::vector<VOISphere>& nset);
    void clear_nodule();

    const std::vector<VOISphere>& get_nodule_set() const;
    void get_nodule_set(std::vector<VOISphere>& nset) const;

private:
    std::vector<VOISphere> _nodule_set;
};

MED_IMG_END_NAMESPACE

#endif