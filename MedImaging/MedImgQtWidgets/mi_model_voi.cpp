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
    _voi_sphere_list.push_back(voi);
    set_changed();
}

void VOIModel::remove_voi_sphere(int id)
{
    int delete_id = 0;
    for (auto it = _voi_sphere_list.begin() ; it != _voi_sphere_list.end() ; ++it)
    {
        if (delete_id++ == id)
        {
            _voi_sphere_list.erase(it);
            set_changed();
            break;
        }
    }
}

const std::list<VOISphere>& VOIModel::get_voi_spheres() const
{
    return _voi_sphere_list;
}

void VOIModel::get_voi_spheres(std::list<VOISphere>& l) const
{
    l = _voi_sphere_list;
}

void VOIModel::modify_voi_sphere_list_rear(const VOISphere& voi)
{
    if (!_voi_sphere_list.empty())
    {
        (--_voi_sphere_list.end())->diameter = voi.diameter;
        (--_voi_sphere_list.end())->center = voi.center;
    }
    set_changed();
}

void VOIModel::modify_voi_sphere_name(int id , std::string name)
{
    int i = 0;
    for (auto it  = _voi_sphere_list.begin() ; it != _voi_sphere_list.end() ; ++it)
    {
        if (i++== id)
        {
            it->name = name;
            return;
        }
    }

    //Find no
}

void VOIModel::modify_voi_sphere_diameter(int id , double diameter)
{
    int i = 0;
    for (auto it  = _voi_sphere_list.begin() ; it != _voi_sphere_list.end() ; ++it)
    {
        if (i++ == id)
        {
            it->diameter= diameter;
            set_changed();
            return;
        }
    }
    //Find no
}

void VOIModel::modify_voi_sphere_center(int id , const Point3& center)
{
    int i = 0;
    for (auto it  = _voi_sphere_list.begin() ; it != _voi_sphere_list.end() ; ++it)
    {
        if (i++ == id)
        {
            it->center = center;
            set_changed();
            return;
        }
    }
}

VOISphere VOIModel::get_voi_sphere(int id)
{
    int i = 0;
    for (auto it  = _voi_sphere_list.begin() ; it != _voi_sphere_list.end() ; ++it)
    {
        if (i++ == id)
        {
            return *it;
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Get voi sphere failed!");
}

void VOIModel::remove_voi_sphere_list_rear()
{
    if (!_voi_sphere_list.empty())
    {
        _voi_sphere_list.erase(--_voi_sphere_list.end());
        set_changed();
    }
}

void VOIModel::remove_all()
{
    std::list<VOISphere>().swap(_voi_sphere_list);
    set_changed();
}

MED_IMAGING_END_NAMESPACE
