#include "mi_model_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

VOIModel::VOIModel()
{

}

VOIModel::~VOIModel()
{

}

void VOIModel::AddVOISphere(const VOISphere& voi)
{
    m_VOISphereList.push_back(voi);
    SetChanged();
}

void VOIModel::RemoveVOISphere(int id)
{
    int iDelete = 0;
    for (auto it = m_VOISphereList.begin() ; it != m_VOISphereList.end() ; ++it)
    {
        if (iDelete == id)
        {
            m_VOISphereList.erase(it);
            break;
        }
    }
}

const std::list<VOISphere>& VOIModel::GetVOISpheres() const
{
    return m_VOISphereList;
}

void VOIModel::GetVOISpheres(std::list<VOISphere>& l) const
{
    l = m_VOISphereList;
}

void VOIModel::ModifyVOISphereRear(const VOISphere& voi)
{
    if (!m_VOISphereList.empty())
    {
        (--m_VOISphereList.end())->m_dDiameter = voi.m_dDiameter;
        (--m_VOISphereList.end())->m_ptCenter = voi.m_ptCenter;
    }
    SetChanged();
}

void VOIModel::ModifyVOISphereName(int id , std::string sName)
{
    int i = 0;
    for (auto it  = m_VOISphereList.begin() ; it != m_VOISphereList.end() ; ++it)
    {
        if (i++== id)
        {
            it->m_sName = sName;
            return;
        }
    }

    //Find no
}

void VOIModel::ModifyVOISphereDiameter(int id , double dDiameter)
{
    int i = 0;
    for (auto it  = m_VOISphereList.begin() ; it != m_VOISphereList.end() ; ++it)
    {
        if (i++ == id)
        {
            it->m_dDiameter= dDiameter;
            SetChanged();
            return;
        }
    }
    //Find no
}

VOISphere VOIModel::GetVOISphere(int id)
{
    int i = 0;
    for (auto it  = m_VOISphereList.begin() ; it != m_VOISphereList.end() ; ++it)
    {
        if (i++ == id)
        {
            return *it;
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Get voi sphere failed!");
}

MED_IMAGING_END_NAMESPACE
