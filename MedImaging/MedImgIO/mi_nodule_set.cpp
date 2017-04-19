#include "mi_nodule_set.h"

MED_IMAGING_BEGIN_NAMESPACE

NoduleSet::NoduleSet()
{

}

NoduleSet::~NoduleSet()
{

}

void NoduleSet::AddNodule(const VOISphere& v)
{
    m_vecNoduleSet.push_back(v);
}

const std::vector<VOISphere>& NoduleSet::GetNoduleSet() const
{
    return m_vecNoduleSet;
}

void NoduleSet::GetNoduleSet(std::vector<VOISphere>& nset) const
{
    nset = m_vecNoduleSet;
}

MED_IMAGING_END_NAMESPACE