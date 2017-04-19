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
    void AddNodule(const VOISphere& v);
    const std::vector<VOISphere>& GetNoduleSet() const;
    void GetNoduleSet(std::vector<VOISphere>& nset) const;

private:
    std::vector<VOISphere> m_vecNoduleSet;
};

MED_IMAGING_END_NAMESPACE


#endif