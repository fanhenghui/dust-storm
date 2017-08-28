#ifndef MED_IMG_VR_SCENE_H
#define MED_IMG_VR_SCENE_H

#include "renderalgo/mi_ray_cast_scene.h"
#include "renderalgo/mi_camera_calculator.h"
#include "arithmetic/mi_plane.h"
#include "arithmetic/mi_aabb.h"

#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_opacity_transfer_function.h"

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

    virtual void set_window_level(float ww , float wl , unsigned char label);
    virtual void set_global_window_level(float ww , float wl);

    void set_bounding_box(const AABB& aabb);

    void set_clipping_plane(std::vector<Plane> planes);

    //void place_vr(ScanSliceType type , bool positive); TODO

    void set_proxy_geometry(ProxyGeometry pg_type);

protected:
    virtual void pre_render_i();
private:
};

MED_IMG_END_NAMESPACE

#endif