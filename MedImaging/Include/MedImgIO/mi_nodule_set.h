#ifndef MED_IMAGING_NODULE_SET_H_
#define MED_IMAGING_NODULE_SET_H_

#include <vector>
#include "MedImgIO/mi_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

class IO_Export NoduleSet
{
public:
    NoduleSet();
    ~NoduleSet();

    void add_nodule(const VOISphere& v);
    void clear_nodule();

    const std::vector<VOISphere>& get_nodule_set() const;
    void get_nodule_set(std::vector<VOISphere>& nset) const;

private:
    std::vector<VOISphere> nodule_set;
};

MED_IMAGING_END_NAMESPACE


#endif