#include "mi_model_cross_hair.h"
#include "MedImgArithmetic/mi_intersection_test.h"
#include "MedImgArithmetic/mi_camera_base.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

MED_IMAGING_BEGIN_NAMESPACE

CrosshairModel::CrosshairModel():m_iForceID(0),m_bVisible(true)
{
    m_aPage[0] = 1;
    m_aPage[1] = 1;
    m_aPage[2] = 1;
}

CrosshairModel::~CrosshairModel()
{

}

void CrosshairModel::set_mpr_scene(const ScanSliceType (&aScanType)[3] ,const MPRScenePtr (&aMPRScenes)[3] ,const RGBUnit (aMPRColors)[3])
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(aMPRScenes[0]);
    m_pCameraCal = aMPRScenes[0]->get_camera_calculator();

    for (int i = 0; i<3 ; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(aMPRScenes[i]);
        m_aMPRScene[i] = aMPRScenes[i];
        m_aMPRColor[i] = aMPRColors[i];
        m_aPage[i] =  m_pCameraCal->get_default_page(aScanType[i]);
    }

    m_ptLocationDiscreteW = m_pCameraCal->get_default_mpr_center_world();
    m_ptLocationContineousW = m_ptLocationDiscreteW;
}

void CrosshairModel::get_cross_line(const MPRScenePtr& pTargetMPRScene, Line2D (&lines)[2] , RGBUnit (&color)[2])
{
    //1 Choose crossed MPR
    QTWIDGETS_CHECK_NULL_EXCEPTION(pTargetMPRScene);
    MPRScenePtr aCrossScene[2] = {nullptr , nullptr};
    int id = 0;
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            continue;
        }
        aCrossScene[id] = m_aMPRScene[i];
        color[id++] = m_aMPRColor[i];
    }
    assert(id == 2);


    //2 MPR plane intersected to a plane
    const Matrix4 matVP = pTargetMPRScene->get_camera()->get_view_projection_matrix();
    Plane planeTarget = pTargetMPRScene->to_plane();
    for (int i = 0; i<2; ++i)
    {
        Plane p = aCrossScene[i]->to_plane();
        Line3D lineIntersect;
        if( IntersectionTest::plane_to_plane(p, planeTarget,lineIntersect))
        {
            //Project intersected line to screen
            Point3 ptScreen = matVP.transform(lineIntersect._pt);
            lines[i]._pt = Point2(ptScreen.x , ptScreen.y);
            Vector3 vDir = matVP.get_inverse().get_transpose().transform(lineIntersect._dir);
            lines[i]._dir = Vector2(vDir.x , vDir.y).get_normalize();
        }
        else
        {
            lines[i]._pt = Point2::S_ZERO_POINT;
            lines[i]._dir = Vector2(0,0);
        }
    }
}

RGBUnit CrosshairModel::get_border_color(MPRScenePtr pTargetMPRScene)
{
    for (int i = 0 ; i< 3 ; ++i)
    {
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            return m_aMPRColor[i];
        }
    }
    return RGBUnit();
}

bool CrosshairModel::page_to(const std::shared_ptr<MPRScene>& pTargetMPRScene, int iPage)
{
    //1 page target MPR
    int iCurrentPage= get_page(pTargetMPRScene);
    if (iCurrentPage == iPage)
    {
        return false;
    }

    std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(pTargetMPRScene->get_camera());
    if( !m_pCameraCal->page_orthognal_mpr_to(pCamera , iPage))
    {
        return false;
    }

    pTargetMPRScene->set_dirty(true);
    set_page_i(pTargetMPRScene , iPage);

    //2 Change cross location
    const Point3 ptCenter = pTargetMPRScene->get_camera()->get_look_at();
    const Vector3 vDir = pTargetMPRScene->get_camera()->get_view_direction();
    const double dDistance = vDir.dot_product(ptCenter - m_ptLocationContineousW);
    m_ptLocationContineousW += dDistance*vDir;
    m_ptLocationDiscreteW += dDistance*vDir;

    set_changed();

    return true;
}

bool CrosshairModel::page(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPageStep)
{
    //1 page target MPR
    std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(pTargetMPRScene->get_camera());
    if( !m_pCameraCal->page_orthognal_mpr(pCamera , iPageStep))
    {
        return false;
    }

    pTargetMPRScene->set_dirty(true);
    set_page_i(pTargetMPRScene , m_pCameraCal->get_orthognal_mpr_page(pCamera));

    //2 Change cross location
    const Point3 ptCenter = pTargetMPRScene->get_camera()->get_look_at();
    const Vector3 vDir = pTargetMPRScene->get_camera()->get_view_direction();
    const double dDistance = vDir.dot_product(ptCenter - m_ptLocationContineousW);
    m_ptLocationContineousW += dDistance*vDir;
    m_ptLocationDiscreteW += dDistance*vDir;

    set_changed();

    return true;
}

bool CrosshairModel::locate(const std::shared_ptr<MPRScene>& pTargetMPRScene , const Point2& pt_dc)
{
    //1 Get latest location
    Point3 ptV;
    if (!pTargetMPRScene->get_volume_position(pt_dc , ptV))
    {
        return false;
    }

    const Matrix4 matV2W = m_pCameraCal->get_volume_to_world_matrix();
    m_ptLocationContineousW = matV2W.transform(ptV);
    m_ptLocationDiscreteW = matV2W.transform(Point3( (double)( (int)ptV.x) , (double)( (int)ptV.y) ,(double)( (int)ptV.z) ));

    //2 Choose crossed MPR
    QTWIDGETS_CHECK_NULL_EXCEPTION(pTargetMPRScene);
    MPRScenePtr aCrossScene[2] = {nullptr , nullptr};
    int id = 0;
    int aIdx[2] = {0,0};
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            continue;
        }
        aCrossScene[id] = m_aMPRScene[i];
        aIdx[id++]  = i;
    }
    assert(id == 2);

    //3 Translate crossed MPR( update LookAt and update Page)
    for (int i = 0; i<2 ; ++i)
    {
        std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(aCrossScene[i]->get_camera());
        m_pCameraCal->translate_mpr_to(pCamera, m_ptLocationContineousW);

        aCrossScene[i]->set_dirty(true);
        int iPage = m_pCameraCal->get_orthognal_mpr_page(pCamera);
        m_aPage[aIdx[i]] = iPage;
    }

    set_changed();

    return true;

}

bool CrosshairModel::locate(const Point3& ptCenterW)
{
    //3 MPR plane paging to the input point slice towards to each normal
    //don't focus the center
    if (!set_center_i(ptCenterW))
    {
        return false;
    }

    for (int i = 0 ; i<3 ; ++ i)
    {
        std::shared_ptr<OrthoCamera> pCamera = std::dynamic_pointer_cast<OrthoCamera>(m_aMPRScene[i]->get_camera());
        m_pCameraCal->translate_mpr_to(pCamera, m_ptLocationContineousW);

        m_aMPRScene[i]->set_dirty(true);
        int iPage = m_pCameraCal->get_orthognal_mpr_page(pCamera);
        m_aPage[i] = iPage;
    }

    set_changed();

    return true;
}

bool CrosshairModel::locate_focus(const Point3& ptCenterW)
{
    //Place MPR center to this center

    return true;
}

void CrosshairModel::set_page_i(const std::shared_ptr<MPRScene>& pTargetMPRScene , int iPage)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            m_aPage[i] = iPage;
            return;
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Cant find certain MPR scene!");
}

int CrosshairModel::get_page(const std::shared_ptr<MPRScene>& pTargetMPRScene)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            return m_aPage[i];
        }
    }
    QTWIDGETS_THROW_EXCEPTION("Cant find certain MPR scene!");
}

bool CrosshairModel::set_center_i(const Point3& ptCenterW)
{
    QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[0]);
    std::shared_ptr<VolumeInfos> pVolumeInfos = m_aMPRScene[0]->get_volume_infos();
    QTWIDGETS_CHECK_NULL_EXCEPTION(pVolumeInfos);
    std::shared_ptr<ImageData> pVolume = pVolumeInfos->get_volume();
    QTWIDGETS_CHECK_NULL_EXCEPTION(pVolume);
    unsigned int *uiDim = pVolume->m_uiDim;

    Point3 ptV = m_pCameraCal->get_world_to_volume_matrix().transform(ptCenterW);
    if (!ArithmeticUtils::check_in_bound(ptV , Point3(uiDim[0] , uiDim[1] , uiDim[2])))
    {
        return false;
    }

    m_ptLocationContineousW = ptCenterW;
    
    m_ptLocationDiscreteW = m_pCameraCal->get_volume_to_world_matrix().transform(
        Point3(double((int)ptV.x) , double((int)ptV.y) , double((int)ptV.z) ));

    return true;
}

bool CrosshairModel::check_focus(MPRScenePtr pTargetMPRScene)
{
    for (int i = 0 ; i< 3; ++i)
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
        if (m_aMPRScene[i] == pTargetMPRScene)
        {
            if(m_iForceID == i)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    return false;
}

void CrosshairModel::focus(MPRScenePtr pTargetMPRScene)
{
    if (!pTargetMPRScene)
    {
        m_iForceID = -1;
    }
    else
    {
        for (int i = 0 ; i< 3; ++i)
        {
            QTWIDGETS_CHECK_NULL_EXCEPTION(m_aMPRScene[i]);
            if (m_aMPRScene[i] == pTargetMPRScene)
            {
                m_iForceID = i;
                break;
            }
        }
    }
}

void CrosshairModel::set_visibility(bool bFlag)
{
    m_bVisible = bFlag;
}

bool CrosshairModel::get_visibility() const
{
    return m_bVisible;
}





MED_IMAGING_END_NAMESPACE