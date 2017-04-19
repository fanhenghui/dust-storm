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

class QtWidgets_Export CrosshairModel : public MedImaging::IModel
{
public:
    typedef std::shared_ptr<MPRScene> MPRScenePtr;

    CrosshairModel();

    virtual ~CrosshairModel();

    void SetMPRScene(const ScanSliceType (&aScanType)[3] , const MPRScenePtr (&aMPRScenes)[3] , const RGBUnit (aMPRColors)[3]);

    void GetCrossLine(
        const MPRScenePtr& pTragetMPRScene, 
        Line2D (&lines)[2],
        RGBUnit (&color)[2]);

    RGBUnit GetBorderColor(MPRScenePtr pTargetMPRScene);

    bool CheckFocus(MPRScenePtr pTargetMPRScene);

    void Focus(MPRScenePtr pTargetMPRScene);

    Point3 GetCrossLocationDiscreteWorld() const;

    Point3 GetCrossLocationContineousWorld() const;

    //Paging one MPR will change cross line in other 2
    bool PagingTo(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPage);

    bool Paging(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPageStep);

    int GetPage(const std::shared_ptr<MPRScene>& pTargetMPRScene );

    //Locate in one MPR will paging others 2
    bool Locate(const std::shared_ptr<MPRScene>& pTargetMPRScene , const Point2& ptDC);

    bool Locate(const Point3& ptCenterW);

    bool LocateFocus(const Point3& ptCenterW);

    void SetVisibility(bool bFlag);

    bool GetVisibility() const;

private:
    void SetPage_i(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPage);

    bool SetCenter_i(const Point3& ptCenterW);

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