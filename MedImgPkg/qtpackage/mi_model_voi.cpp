#include "mi_model_voi.h"
#include "mi_qt_package_logger.h"

MED_IMG_BEGIN_NAMESPACE

VOIModel::VOIModel():_voi_mask_visible(true), _voi_to_be_tuned(-1), _tune_radius(5.0)
{
    this->_voxel_to_be_tuned.reserve(3);
}

VOIModel::~VOIModel()
{

}

void VOIModel::add_voi(const VOISphere& voi , unsigned char label)
{
    _vois.push_back(voi);

    _intensity_infos.push_back(IntensityInfo());

    _labels.push_back(label);

    set_changed();
}

void VOIModel::remove_voi(int id)
{
    if (id < _vois.size())
    {
        auto it = _vois.begin();
        auto it2 = _intensity_infos.begin();
        auto it3 = _labels.begin();
        int i = 0;
        while(i != id)
        {
            ++it;
            ++it2;
            ++it3;
            ++i;
        }

        _vois.erase(it);
        _intensity_infos.erase(it2);
        _labels.erase(it3);
        set_changed();
    }
}

void VOIModel::remove_voi_list_rear()
{
    if (!_vois.empty())
    {
        _vois.erase(--_vois.end());
        _intensity_infos.erase(--_intensity_infos.end());
        _labels.erase(--_labels.end());
        set_changed();
    }
}

void VOIModel::remove_all()
{
    std::vector<VOISphere>().swap(_vois);
    std::vector<IntensityInfo>().swap(_intensity_infos);
    std::vector<unsigned char>().swap(_labels);
    set_changed();
}

const std::vector<VOISphere>& VOIModel::get_vois() const
{
    return _vois;
}

VOISphere VOIModel::get_voi(int id) const
{
    if (id < _vois.size())
    {
        return _vois[id];
    }
    else
    {
        QTWIDGETS_THROW_EXCEPTION("Get voi failed!");
    }
}

unsigned char VOIModel::get_label(int id) const
{
    if (id < _labels.size())
    {
        return _labels[id];
    }
    else
    {
        QTWIDGETS_THROW_EXCEPTION("Get voi label failed!");
    }
}

const std::vector<unsigned char>& VOIModel::get_labels() const
{
    return _labels;
}

void VOIModel::modify_voi_list_rear(const VOISphere& voi)
{
    if (!_vois.empty())
    {
        _vois[_vois.size() - 1].diameter = voi.diameter;
        _vois[_vois.size() - 1].center = voi.center;
    }
    set_changed();
}

void VOIModel::set_tune_radius(const double r)
{
    if (r > 0.0)
    {
        this->_tune_radius = r;
    }
}

double VOIModel::get_tune_radius()
{
    return this->_tune_radius;
}

void VOIModel::modify_name(int id , std::string name)
{
    if (id < _vois.size())
    {
        _vois[id].name = name;
    }

    //Find no
}

void VOIModel::modify_diameter(int id , double diameter)
{
    if (id < _vois.size())
    {
        _vois[id].diameter = diameter;
        set_changed();
    }
    //Find no
}

void VOIModel::modify_center(int id , const Point3& center)
{
    if (id < _vois.size())
    {
        _vois[id].center = center;
        set_changed();
    }
}

void VOIModel::modify_intensity_info(int id , IntensityInfo info)
{
    if (id < _intensity_infos.size())
    {
        _intensity_infos[id] = info;
    }
}


IntensityInfo VOIModel::get_intensity_info(int id)
{
    if (id < _intensity_infos.size())
    {
        return _intensity_infos[id];
    }
    else
    {
        QTWIDGETS_THROW_EXCEPTION("Get voi intensity info failed!");
    }
}

int VOIModel::get_voi_number() const
{
    return _vois.size();
}

const std::vector<IntensityInfo>& VOIModel::get_intensity_infos() const
{
    return _intensity_infos;
}

void VOIModel::set_voi_to_tune(int voi_idx)
{
    this->_voi_to_be_tuned = voi_idx;
}

int VOIModel::get_voi_to_tune()
{
    return this->_voi_to_be_tuned;
}

void VOIModel::set_tune_location(const Point3& loc)
{
    this->_tune_location = loc;
}
const Point3& VOIModel::get_tune_location()
{
    return this->_tune_location;
}

//void VOIModel::set_voxel_block_to_tune(AABBUI& voxel_range, int /*tune_type*/)
//{
//    this->_voxel_block_to_be_tuned = voxel_range;
//    this->set_changed();
//}
//
//const AABBUI& VOIModel::get_voxel_block_to_tune()
//{
//    return this->_voxel_block_to_be_tuned;
//}

void VOIModel::set_voxel_to_tune(const std::vector<unsigned int>& voxel_idx, int tune_type)
{
    if (voxel_idx.size() == 3)
    {
        this->_voxel_to_be_tuned = voxel_idx;
        this->set_changed();
    }
}

const std::vector<unsigned int>& VOIModel::get_voxel_to_tune()
{
    return this->_voxel_to_be_tuned;
}

bool VOIModel::is_voi_mask_visible() const
{
    return _voi_mask_visible;
}

void VOIModel::set_voi_mask_visible(bool flag)
{
    _voi_mask_visible = flag;
}

VOISphere VOIModel::get_voi_by_label(unsigned char id) const
{
    for (int i = 0 ; i<_labels.size() ; ++i)
    {
        if (id == _labels[i])
        {
            return _vois[i];
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Get voi by mask label failed!");
}

void VOIModel::print_code_id(int code_id)
{
    switch (code_id)
    {
    case MODIFYING:
        {
            MI_QTPACKAGE_LOG(MI_DEBUG) << "VOI model modifying.";
            break;
        }
    case MODIFY_COMPLETED:
        {
            MI_QTPACKAGE_LOG(MI_DEBUG) << "VOI model modify completed.";
            break;
        }
    case ADD_VOI:
        {
            MI_QTPACKAGE_LOG(MI_DEBUG) << "VOI model add voi.";
            break;
        }
    case DELETE_VOI:
        {
            MI_QTPACKAGE_LOG(MI_DEBUG) << "VOI model delete voi.";
            break;
        }
    default:
        {
            MI_QTPACKAGE_LOG(MI_DEBUG) << "VOI model Unknown code id.";
        }
    }
}

MED_IMG_END_NAMESPACE
