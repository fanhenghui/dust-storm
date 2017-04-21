#include "mi_mpr_scene.h"

#include "MedImgCommon/mi_configuration.h"

#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"

#include "mi_volume_infos.h"



MED_IMAGING_BEGIN_NAMESPACE

MPRScene::MPRScene():RayCastScene()
{
    std::shared_ptr<MPREntryExitPoints> pMPREE(new MPREntryExitPoints());
    _entry_exit_points = pMPREE;
    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _entry_exit_points->set_strategy(CPU_BASE);
    }
    else
    {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

MPRScene::MPRScene(int width , int height):RayCastScene(width , height)
{
    std::shared_ptr<MPREntryExitPoints> pMPREE(new MPREntryExitPoints());
    _entry_exit_points = pMPREE;
    if (CPU == Configuration::instance()->get_processing_unit_type())
    {
        _entry_exit_points->set_strategy(CPU_BASE);
    }
    else
    {
        _entry_exit_points->set_strategy(GPU_BASE);
    }
}

MPRScene::~MPRScene()
{

}

void MPRScene::place_mpr(ScanSliceType eType)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_camera_calculator);
    //Calculate MPR placement camera
    _camera_calculator->init_mpr_placement(m_pRayCastCamera , eType);
    //Set initial camera to interactor
    m_pCameraInteractor->set_initial_status(m_pRayCastCamera);
    //resize because initial camera's ratio between width and height  is 1, but current ratio may not.
    m_pCameraInteractor->resize(_width , _height);

    set_dirty(true);
}

void MPRScene::rotate(const Point2& pre_pt , const Point2& cur_pt)
{
    m_pCameraInteractor->rotate(pre_pt , cur_pt , _width , _height );
    set_dirty(true);
}

void MPRScene::zoom(const Point2& pre_pt , const Point2& cur_pt)
{
    m_pCameraInteractor->zoom(pre_pt , cur_pt , _width , _height );
    set_dirty(true);
}

void MPRScene::pan(const Point2& pre_pt , const Point2& cur_pt)
{
    m_pCameraInteractor->pan(pre_pt , cur_pt , _width , _height );
    set_dirty(true);
}

bool MPRScene::get_volume_position(const Point2& pt_dc , Point3& ptPosV)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(m_pVolumeInfos);
    std::shared_ptr<ImageData> pImg = m_pVolumeInfos->get_volume();
    RENDERALGO_CHECK_NULL_EXCEPTION(pImg);

    Point2 pt = ArithmeticUtils::dc_to_ndc(pt_dc , _width , _height);

    Matrix4 mat_mvp = m_pRayCastCamera->get_view_projection_matrix()*_camera_calculator->get_volume_to_world_matrix();
    mat_mvp.inverse();

    Point3 ptVolume = mat_mvp.transform(Point3(pt.x , pt.y , 0.0));
    if (ArithmeticUtils::check_in_bound(ptVolume , Point3(pImg->_dim[0] - 1.0 , pImg->_dim[1] - 1 , pImg->_dim[2] - 1)))
    {
        ptPosV = ptVolume;
        return true;
    }
    else
    {
        return false;
    }
}

bool MPRScene::get_world_position(const Point2& pt_dc , Point3& ptPosW)
{
    Point3 ptPosV;
    if (get_volume_position(pt_dc , ptPosV))
    {
        ptPosW = _camera_calculator->get_volume_to_world_matrix().transform(ptPosV);
        return true;
    }
    else
    {
        return false;
    }
}

void MPRScene::page(int iStep)
{
    //TODO should consider oblique MPR
    _camera_calculator->page_orthognal_mpr(m_pRayCastCamera , iStep);
    set_dirty(true);
}

void MPRScene::page_to(int page)
{
    _camera_calculator->page_orthognal_mpr_to(m_pRayCastCamera , page);
    set_dirty(true);
}

Plane MPRScene::to_plane() const
{
    Point3 ptEye = m_pRayCastCamera->get_eye();
    Point3 ptLookAt = m_pRayCastCamera->get_look_at();

    Vector3 vNorm = ptLookAt - ptEye;
    vNorm.normalize();

    Plane p;
    p._norm = vNorm;
    p._distance = vNorm.dot_product(ptLookAt - Point3::S_ZERO_POINT);

    return p;
}

bool MPRScene::get_patient_position(const Point2& pt_dc, Point3& ptPosP)
{
    Point3 ptW;
    if (get_world_position(pt_dc , ptW))
    {
        ptPosP = _camera_calculator->get_world_to_patient_matrix().transform(ptW);
        return true;
    }
    else
    {
        return false;
    }
}

MED_IMAGING_END_NAMESPACE
