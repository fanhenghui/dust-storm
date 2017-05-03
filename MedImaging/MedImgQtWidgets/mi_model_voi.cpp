#include "mi_model_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

VOIModel::VOIModel():_focus_id(-1)
{

}

VOIModel::~VOIModel()
{

}

void VOIModel::add_voi_sphere(const VOISphere& voi)
{
    _vois.push_back(voi);
    _voi_intensity_infos.push_back(IntensityInfo());
    _voi_intensity_infos_dirty.push_back(false);
    set_changed();
}

void VOIModel::remove_voi_sphere(int id)
{
    if (id < _vois.size())
    {
        auto it = _vois.begin();
        auto it2 = _voi_intensity_infos.begin();
        int i = 0;
        while(i != id)
        {
            ++it;
            ++it2;
            ++i;
        }

        _vois.erase(it);
        _voi_intensity_infos.erase(it2);
        set_changed();
    }
}

const std::vector<VOISphere>& VOIModel::get_voi_spheres() const
{
    return _vois;
}

void VOIModel::get_voi_spheres(std::vector<VOISphere>& l) const
{
    l = _vois;
}

void VOIModel::modify_voi_sphere_list_rear(const VOISphere& voi)
{
    if (!_vois.empty())
    {
        _vois[_vois.size() - 1].diameter = voi.diameter;
        _vois[_vois.size() - 1].center = voi.center;
    }
    set_changed();
}

void VOIModel::modify_voi_sphere_name(int id , std::string name)
{
    if (id < _vois.size())
    {
        _vois[id].name = name;
    }

    //Find no
}

void VOIModel::modify_voi_sphere_diameter(int id , double diameter)
{
    if (id < _vois.size())
    {
        _vois[id].diameter = diameter;
        set_changed();
    }
    //Find no
}

void VOIModel::modify_voi_sphere_center(int id , const Point3& center)
{
    if (id < _vois.size())
    {
        _vois[id].center = center;
        set_changed();
    }
}

VOISphere VOIModel::get_voi_sphere(int id)
{
    if (id < _vois.size())
    {
        auto it = _vois.begin();
        int i = 0;
        while(i != id)
        {
            ++it;
            ++i;
        }
        return *it;
    }
    else
    {
        QTWIDGETS_THROW_EXCEPTION("Get voi sphere failed!");
    }
}

void VOIModel::modify_voi_sphere_intensity_info(int id , IntensityInfo info)
{
    if (id < _voi_intensity_infos.size())
    {
        _voi_intensity_infos[id] = info;
    }
}

bool VOIModel::is_voi_sphere_intensity_info_dirty(int id)
{
    if (id < _voi_intensity_infos_dirty.size())
    {
        return _voi_intensity_infos_dirty[id];
    }
    else
    {
        QTWIDGETS_THROW_EXCEPTION("Get voi sphere failed!");
    }
}

IntensityInfo VOIModel::get_voi_sphere_intensity_info(int id)
{
    if (id < _voi_intensity_infos.size())
    {
        return _voi_intensity_infos[id];
    }
    else
    {
        QTWIDGETS_THROW_EXCEPTION("Get voi sphere intensity info failed!");
    }
}


void VOIModel::remove_voi_sphere_list_rear()
{
    if (!_vois.empty())
    {
        _vois.erase(--_vois.end());
        _voi_intensity_infos.erase(--_voi_intensity_infos.end());
        set_changed();
    }
}

void VOIModel::remove_all()
{
    std::vector<VOISphere>().swap(_vois);
    std::vector<IntensityInfo>().swap(_voi_intensity_infos);
    set_changed();
}

int VOIModel::get_voi_number() const
{
    return _vois.size();
}

void VOIModel::set_voi_sphere_intensity_info_dirty(int id , bool flag)
{
    if (id < _voi_intensity_infos_dirty.size())
    {
        _voi_intensity_infos_dirty[id] = flag;
    }
}

const std::vector<IntensityInfo>& VOIModel::get_voi_sphere_intensity_infos() const
{
    return _voi_intensity_infos;
}

//void VOIModel::focus(int id)
//{
//    _focus_id = id;
//}
//
//int VOIModel::get_focus() const
//{
//    return _focus_id;
//}



MED_IMAGING_END_NAMESPACE
