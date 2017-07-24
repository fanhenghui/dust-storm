#ifndef MED_IMAGING_MODEL_VOI_H_
#define MED_IMAGING_MODEL_VOI_H_

#include <vector>
#include "MedImgCommon/mi_model_interface.h"
#include "MedImgIO/mi_voi.h"
#include "MedImgArithmetic/mi_volume_statistician.h"
#include "MedImgArithmetic/mi_aabb.h"

MED_IMAGING_BEGIN_NAMESPACE


//Notify code ID:
//0 修改结束 
//1 修改过程中
//2 添加voi
//3 删除voi
class QtWidgets_Export VOIModel : public IModel
{
public:
    enum CodeID
    {
        MODIFY_COMPLETED = 0,
        MODIFYING = 1, 
        ADD_VOI = 2, // add voi by drawing a sphere
        DELETE_VOI = 3, // delete the voi
        TUNING_VOI = 4, // tune voi (aka eraser) specified by user
        LOAD_VOI = 5, // load/read voi (, and reconstruct the sphere)
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

    void set_voi_to_tune(int voi_idx);
    int get_voi_to_tune();
    void set_tune_radius(const double r);
    double get_tune_radius();
    void set_tune_location(const Point3& loc);
    const Point3& get_tune_location();
    
    //void set_voxel_block_to_tune(AABBUI& voxel_range, int tune_type);
    //const AABBUI& get_voxel_block_to_tune();

    void set_voxel_to_tune(const std::vector<unsigned int>& voxel_idx, int tune_type);
    const std::vector<unsigned int>& get_voxel_to_tune();

    bool is_voi_mask_visible() const;
    void set_voi_mask_visible(bool flag);

protected:
private:
    std::vector<VOISphere>       _vois;
    std::vector<unsigned char>   _labels;
    std::vector<IntensityInfo>   _intensity_infos;
    bool _voi_mask_visible;

    // which voi to tune, which voxels to tune
    int _voi_to_be_tuned;
    //AABBUI _voxel_block_to_be_tuned;
    std::vector<unsigned int> _voxel_to_be_tuned;
    double _tune_radius;
    Point3 _tune_location; 
};

MED_IMAGING_END_NAMESPACE

#endif