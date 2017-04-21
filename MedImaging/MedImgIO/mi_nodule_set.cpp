#include "mi_nodule_set.h"

MED_IMAGING_BEGIN_NAMESPACE

NoduleSet::NoduleSet()
{

}

NoduleSet::~NoduleSet()
{

}

void NoduleSet::add_nodule(const VOISphere& v)
{
    nodule_set.push_back(v);
}

const std::vector<VOISphere>& NoduleSet::get_nodule_set() const
{
    return nodule_set;
}

void NoduleSet::get_nodule_set(std::vector<VOISphere>& nset) const
{
    nset = nodule_set;
}

MED_IMAGING_END_NAMESPACE