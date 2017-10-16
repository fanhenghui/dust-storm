#ifndef MED_IMG_APPCOMMON_MI_MODEL_CROSSHAIR_H
#define MED_IMG_APPCOMMON_MI_MODEL_CROSSHAIR_H

#include "appcommon/mi_app_common_export.h"
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

class AppCommon_Export ModelCrosshair : public IModel {
public:
    typedef std::shared_ptr<MPRScene> MPRScenePtr;

    ModelCrosshair();
    virtual ~ModelCrosshair();

    void set_mpr_scene(const ScanSliceType(&scan_type)[3], const MPRScenePtr(&scenes)[3], const RGBUnit(colors)[3]);

    void set_mpr_group_orthogonality(bool flag);//call when trigger on MPR rotation
    bool get_mpr_group_orthogonality() const;

    void get_cross_line(const MPRScenePtr& target_mpr_scene, Line2D(&lines)[2], RGBUnit(&color)[2]);
    bool get_cross(const MPRScenePtr& target_mpr_scene, Point2& pt_dc);
    RGBUnit get_border_color(MPRScenePtr target_mpr_scene);

    Point3 get_cross_location_discrete_world() const;
    Point3 get_cross_location_contineous_world() const;

    //orthogonal interface
    //page one MPR will change cross line in other 2
    bool page_to(const std::shared_ptr<MPRScene>& target_mpr_scene, int page);
    bool page(const std::shared_ptr<MPRScene>& target_mpr_scene, int step);
    int get_page(const std::shared_ptr<MPRScene>& target_mpr_scene);

    //locate in one MPR will paging others 2
    bool locate(const std::shared_ptr<MPRScene>& target_mpr_scene, const Point2& pt_dc);//2d
    bool locate(const Point3& center_w, bool ignore_pan = true);//3d

    //cross line & crosshair's visibility
    void set_visibility(bool flag);
    bool get_visibility() const;

    void reset();//reset MPRs to orginal orthogonal slice

private:
    void set_page_i(const std::shared_ptr<MPRScene>& target_mpr_scene, int page);
    bool set_center_i(const Point3& center_w);

private:
    MPRScenePtr _mpr_scenes[3];
    ScanSliceType _mpr_slice_type[3];
    RGBUnit _mpr_colors[3];
    int _pages[3];

    Point3 _location_discrete_w;
    Point3 _location_contineous_w;

    std::shared_ptr<CameraCalculator> _camera_calculator;

    bool _orthogonal;

    bool _visible;

private:
    DISALLOW_COPY_AND_ASSIGN(ModelCrosshair);
};

MED_IMG_END_NAMESPACE

#endif