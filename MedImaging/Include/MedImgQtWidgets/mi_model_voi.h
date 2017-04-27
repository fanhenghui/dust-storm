#ifndef MED_IMAGING_MODEL_VOI_H_
#define MED_IMAGING_MODEL_VOI_H_

#include <vector>
#include "MedImgCommon/mi_model_interface.h"
#include "MedImgIO/mi_voi.h"
#include "MedImgArithmetic/mi_volume_statistician.h"

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export VOIModel : public IModel
{
public:
    VOIModel();
    virtual ~VOIModel();

    void add_voi_sphere(const VOISphere& voi);
    VOISphere get_voi_sphere(int id);
    int get_voi_number() const;

    IntensityInfo get_voi_sphere_intensity_info(int id);
    const std::vector<IntensityInfo>& get_voi_sphere_intensity_infos() const;
    bool is_voi_sphere_intensity_info_dirty(int id);
    void set_voi_sphere_intensity_info_dirty(int id , bool flag);
    void modify_voi_sphere_intensity_info(int id , IntensityInfo info);

    void modify_voi_sphere_name(int id , std::string name);
    void modify_voi_sphere_diameter(int id , double diameter);
    void modify_voi_sphere_center(int id , const Point3& center);

    void modify_voi_sphere_list_rear(const VOISphere& voi);

    void remove_voi_sphere_list_rear();
    void remove_voi_sphere(int id);
    void remove_all();

    const std::vector<VOISphere>& get_voi_spheres() const;
    void get_voi_spheres(std::vector<VOISphere>& l) const;

protected:
private:
    std::vector<VOISphere>      _vois;
    std::vector<IntensityInfo>   _voi_intensity_infos;
    std::vector<bool>   _voi_intensity_infos_dirty;
};

MED_IMAGING_END_NAMESPACE

#endif