#ifndef MED_IMAGING_CROSS_HAIR_H_
#define MED_IMAGING_CROSS_HAIR_H_

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "MedImgCommon/mi_model_interface.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgArithmetic/mi_line.h"
#include "MedImgRenderAlgorithm/mi_camera_calculator.h"

MED_IMAGING_BEGIN_NAMESPACE

class MPRScene;
class SceneBase;
class CameraCalculator;

class QtWidgets_Export CrosshairModel : public medical_imaging::IModel
{
public:
    typedef std::shared_ptr<MPRScene> MPRScenePtr;

    CrosshairModel();

    virtual ~CrosshairModel();

    void set_mpr_scene(const ScanSliceType (&aScanType)[3] , const MPRScenePtr (&aMPRScenes)[3] , const RGBUnit (aMPRColors)[3]);

    void get_cross_line(
        const MPRScenePtr& pTragetMPRScene, 
        Line2D (&lines)[2],
        RGBUnit (&color)[2]);

    RGBUnit get_border_color(MPRScenePtr pTargetMPRScene);

    bool check_focus(MPRScenePtr pTargetMPRScene);

    void focus(MPRScenePtr pTargetMPRScene);

    Point3 get_cross_location_discrete_world() const;

    Point3 get_cross_location_contineous_world() const;

    //page one MPR will change cross line in other 2
    bool page_to(const std::shared_ptr<MPRScene>& pTargetMPRScene , int page);

    bool page(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPageStep);

    int get_page(const std::shared_ptr<MPRScene>& pTargetMPRScene );

    //locate in one MPR will paging others 2
    bool locate(const std::shared_ptr<MPRScene>& pTargetMPRScene , const Point2& pt_dc);

    bool locate(const Point3& ptCenterW);

    bool locate_focus(const Point3& ptCenterW);

    void set_visibility(bool flag);

    bool get_visibility() const;

private:
    void set_page_i(const std::shared_ptr<MPRScene>& pTargetMPRScene , int page);

    bool set_center_i(const Point3& ptCenterW);

private:
    MPRScenePtr m_aMPRScene[3];
    RGBUnit m_aMPRColor[3];
    int m_aPage[3];
    int m_iForceID;

    Point3 m_ptLocationDiscreteW;
    Point3 m_ptLocationContineousW;

    std::shared_ptr<CameraCalculator> m_pCameraCal;

    bool m_bVisible;
};

MED_IMAGING_END_NAMESPACE

#endif