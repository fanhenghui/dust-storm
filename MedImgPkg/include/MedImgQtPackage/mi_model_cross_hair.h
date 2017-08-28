#ifndef MED_IMG_CROSS_HAIR_H_
#define MED_IMG_CROSS_HAIR_H_

#include "MedImgQtPackage/mi_qt_package_export.h"
#include "util/mi_model_interface.h"
#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_color_unit.h"
#include "arithmetic/mi_line.h"
#include "renderalgo/mi_camera_calculator.h"

MED_IMG_BEGIN_NAMESPACE

class MPRScene;
class SceneBase;
class CameraCalculator;

class QtPackage_Export CrosshairModel : public medical_imaging::IModel
{
public:
    typedef std::shared_ptr<MPRScene> MPRScenePtr;

    CrosshairModel();

    virtual ~CrosshairModel();

    void set_mpr_scene(const ScanSliceType (&scan_type)[3] , const MPRScenePtr (&scenes)[3] , const RGBUnit (colors)[3]);

    void get_cross_line(
        const MPRScenePtr& target_mpr_scene, 
        Line2D (&lines)[2],
        RGBUnit (&color)[2]);

    RGBUnit get_border_color(MPRScenePtr target_mpr_scene);

    Point3 get_cross_location_discrete_world() const;

    Point3 get_cross_location_contineous_world() const;

    //page one MPR will change cross line in other 2
    bool page_to(const std::shared_ptr<MPRScene>& target_mpr_scene , int page);

    bool page(const std::shared_ptr<MPRScene>& target_mpr_scene , int step);

    int get_page(const std::shared_ptr<MPRScene>& target_mpr_scene );

    //locate in one MPR will paging others 2
    bool locate(const std::shared_ptr<MPRScene>& target_mpr_scene , const Point2& pt_dc);

    bool locate(const Point3& center_w);

    //bool locate_focus(const Point3& center_w);

    void set_visibility(bool flag);

    bool get_visibility() const;

private:
    void set_page_i(const std::shared_ptr<MPRScene>& target_mpr_scene , int page);

    bool set_center_i(const Point3& center_w);

private:
    MPRScenePtr _mpr_scenes[3];
    RGBUnit _mpr_colors[3];
    int _pages[3];

    Point3 _location_discrete_w;
    Point3 _location_contineous_w;

    std::shared_ptr<CameraCalculator> _camera_calculator;

    bool _is_visible;
};

MED_IMG_END_NAMESPACE

#endif