#include "mi_model_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

VOIModel::VOIModel()
{

}

VOIModel::~VOIModel()
{

}

void VOIModel::add_voi_sphere(const VOISphere& voi)
{
    m_VOISphereList.push_back(voi);
    set_changed();
}

void VOIModel::remove_voi_sphere(int id)
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

const std::list<VOISphere>& VOIModel::get_voi_spheres() const
{
    return m_VOISphereList;
}

void VOIModel::get_voi_spheres(std::list<VOISphere>& l) const
{
    l = m_VOISphereList;
}

void VOIModel::modify_voi_sphere_list_rear(const VOISphere& voi)
{
    if (!m_VOISphereList.empty())
    {
        (--m_VOISphereList.end())->m_dDiameter = voi.m_dDiameter;
        (--m_VOISphereList.end())->m_ptCenter = voi.m_ptCenter;
    }
    set_changed();
}

void VOIModel::modify_voi_sphere_name(int id , std::string sName)
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

void VOIModel::modify_voi_sphere_diameter(int id , double dDiameter)
{
    int i = 0;
    for (auto it  = m_VOISphereList.begin() ; it != m_VOISphereList.end() ; ++it)
    {
        if (i++ == id)
        {
            it->m_dDiameter= dDiameter;
            set_changed();
            return;
        }
    }
    //Find no
}

VOISphere VOIModel::get_voi_sphere(int id)
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
