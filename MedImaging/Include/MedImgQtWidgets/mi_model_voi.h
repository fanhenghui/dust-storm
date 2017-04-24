#ifndef MED_IMAGING_MODEL_VOI_H_
#define MED_IMAGING_MODEL_VOI_H_

#include "MedImgCommon/mi_model_interface.h"
#include "MedImgIO/mi_voi.h"
#include <list>

MED_IMAGING_BEGIN_NAMESPACE

class QtWidgets_Export VOIModel : public IModel
{
public:
    VOIModel();
    virtual ~VOIModel();

    void add_voi_sphere(const VOISphere& voi);
    VOISphere get_voi_sphere(int id);

    void modify_voi_sphere_name(int id , std::string name);
    void modify_voi_sphere_diameter(int id , double diameter);

    void modify_voi_sphere_list_rear(const VOISphere& voi);

    void remove_voi_sphere_list_rear();
    void remove_voi_sphere(int id);

    const std::list<VOISphere>& get_voi_spheres() const;
    void get_voi_spheres(std::list<VOISphere>& l) const;

protected:
private:
    std::list<VOISphere> _voi_sphere_list;
};

MED_IMAGING_END_NAMESPACE

#endif