#ifndef MED_IMG_MODEL_VOI_H_
#define MED_IMG_MODEL_VOI_H_

#include <vector>
#include "qtpackage/mi_qt_package_export.h"
#include "util/mi_model_interface.h"
#include "io/mi_voi.h"
#include "arithmetic/mi_volume_statistician.h"

MED_IMG_BEGIN_NAMESPACE 

//Notify code ID:
//0 �޸Ľ��� 
//1 �޸Ĺ�����
//2 ���voi
//3 ɾ��voi
class QtPackage_Export VOIModel : public IModel
{
public:
    enum CodeID
    {
        MODIFY_COMPLETED = 0,
        MODIFYING = 1,
        ADD_VOI = 2,
        DELETE_VOI = 3
    };
    static void print_code_id(int code_id);

    VOIModel();
    virtual ~VOIModel();

    void add_voi(const VOISphere& voi , unsigned char label);
    VOISphere get_voi(int id) const;
    VOISphere get_voi_by_label(unsigned char id) const;
    unsigned char get_label(int id) const;
    const std::vector<unsigned char>& get_labels() const;
    const std::vector<VOISphere>& get_vois() const;

    void remove_voi_list_rear();
    void remove_voi(int id);
    void remove_all();

    int get_voi_number() const;

    IntensityInfo get_intensity_info(int id);
    const std::vector<IntensityInfo>& get_intensity_infos() const;
    void modify_intensity_info(int id , IntensityInfo info);

    void modify_name(int id , std::string name);
    void modify_diameter(int id , double diameter);
    void modify_center(int id , const Point3& center);
    void modify_voi_list_rear(const VOISphere& voi);

    bool is_voi_mask_visible() const;
    void set_voi_mask_visible(bool flag);

protected:
private:
    std::vector<VOISphere>        _vois;
    std::vector<unsigned char>   _labels;
    std::vector<IntensityInfo>     _intensity_infos;
    bool _voi_mask_visible;
};

MED_IMG_END_NAMESPACE

#endif