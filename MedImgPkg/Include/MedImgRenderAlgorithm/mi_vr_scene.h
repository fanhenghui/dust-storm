#ifndef MED_IMG_VR_SCENE_H
#define MED_IMG_VR_SCENE_H

#include "MedImgRenderAlgorithm/mi_ray_cast_scene.h"
#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgArithmetic/mi_plane.h"
#include "MedImgArithmetic/mi_aabb.h"

#include "MedImgRenderAlgorithm/mi_color_transfer_function.h"
#include "MedImgRenderAlgorithm/mi_opacity_transfer_function.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export VRScene : public RayCastScene
{
public:
    VRScene();
    VRScene(int width , int height);
    virtual ~VRScene();

    virtual void rotate(const Point2& pre_pt , const Point2& cur_pt);
    virtual void zoom(const Point2& pre_pt , const Point2& cur_pt);
    virtual void pan(const Point2& pre_pt , const Point2& cur_pt);

    virtual void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);

    void set_bounding_box(const AABB& aabb);

    void set_clipping_plane(std::vector<Plane> planes);

    //void place_vr(ScanSliceType type , bool positive); TODO

    void set_proxy_geometry(ProxyGeometry pg_type);

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif