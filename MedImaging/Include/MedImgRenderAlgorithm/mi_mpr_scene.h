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

    MPRScene(int iWidth , int iHeight);

    virtual ~MPRScene();

    virtual void Rotate(const Point2& ptPre , const Point2& ptCur);

    virtual void Zoom(const Point2& ptPre , const Point2& ptCur);

    virtual void Pan(const Point2& ptPre , const Point2& ptCur);

    //Default positive direction
    // Axial F->H
    // Sagittal R->L
    // Coronal A -> P
    void Paging(int iStep);

    void PagingTo(int iPage);

    //Call to initialize MPR placement
    void PlaceMPR(MedImaging::ScanSliceType eType);

    bool GetVolumePosition(const Point2& ptDC, Point3& ptPosV);

    bool GetWorldPosition(const Point2& ptDC, Point3& ptPosW);

    Plane ToPlane()const;

protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif