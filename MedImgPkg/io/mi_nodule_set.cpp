#include "mi_nodule_set.h"

MED_IMG_BEGIN_NAMESPACE

NoduleSet::NoduleSet()
{

}

NoduleSet::~NoduleSet()
{

}

void NoduleSet::add_nodule(const VOISphere& v)
{
    _nodule_set.push_back(v);
}

const std::vector<VOISphere>& NoduleSet::get_nodule_set() const
{
    return _nodule_set;
}

void NoduleSet::get_nodule_set(std::vector<VOISphere>& nset) const
{
    nset = _nodule_set;
}

void NoduleSet::clear_nodule()
{
    std::vector<VOISphere>().swap(_nodule_set);
}

void NoduleSet::set_nodule(const std::vector<VOISphere>& nset)
{
    _nodule_set = nset;
}

MED_IMG_END_NAMESPACE