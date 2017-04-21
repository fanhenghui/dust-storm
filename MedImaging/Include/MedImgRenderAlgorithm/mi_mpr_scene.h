#ifndef MED_IMAGING_MPR_SCENE_QT_H
#define MED_IMAGING_MPR_SCENE_QT_H

#include "MedImgRenderAlgorithm/mi_ray_cast_scene.h"
#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgArithmetic/mi_plane.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export MPRScene : public RayCastScene
{
public:
    MPRScene();

    MPRScene(int width , int height);

    virtual ~MPRScene();

    virtual void rotate(const Point2& pre_pt , const Point2& cur_pt);

    virtual void zoom(const Point2& pre_pt , const Point2& cur_pt);

    virtual void pan(const Point2& pre_pt , const Point2& cur_pt);

    //Default positive direction
    // Axial F->H
    // Sagittal R->L
    // Coronal A -> P
    void page(int iStep);

    void page_to(int page);

    //Call to initialize MPR placement
    void place_mpr(ScanSliceType eType);

    bool get_volume_position(const Point2& pt_dc, Point3& ptPosV);

    bool get_world_position(const Point2& pt_dc, Point3& ptPosW);

    bool get_patient_position(const Point2& pt_dc, Point3& ptPosP);

    Plane to_plane()const;

protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif