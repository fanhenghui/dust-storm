#ifndef MEDIMGRENDERALGO_VR_SCENE_H
#define MEDIMGRENDERALGO_VR_SCENE_H

#include "arithmetic/mi_aabb.h"
#include "arithmetic/mi_plane.h"
#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_ray_cast_scene.h"

#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_opacity_transfer_function.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export VRScene : public RayCastScene {
public:
    VRScene(RayCastingStrategy strategy, GPUPlatform platfrom);
    VRScene(int width, int height, RayCastingStrategy strategy, GPUPlatform platfrom);
    virtual ~VRScene();

    virtual void initialize();

    virtual void rotate(const Point2& pre_pt, const Point2& cur_pt);
    virtual void zoom(const Point2& pre_pt, const Point2& cur_pt);
    virtual void pan(const Point2& pre_pt, const Point2& cur_pt);

    virtual void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);

    virtual void set_color_opacity(std::shared_ptr<ColorTransFunc> color,
        std::shared_ptr<OpacityTransFunc> opacity,
        unsigned char label);

    void set_bounding_box(const AABB& aabb);

    void set_clipping_plane(std::vector<Plane> planes);

    // void place_vr(ScanSliceType type , bool positive); TODO

    void set_proxy_geometry(ProxyGeometry pg_type);

    virtual void set_visible_labels(std::vector<unsigned char> labels);

    bool get_ray_end(const Point2& pt_cross, Point3& pt_ray_end_world);
    void cache_ray_end();

protected:
    virtual void pre_render();

private:
    struct RayEnd;
    std::unique_ptr<RayEnd> _cache_ray_end;
};

MED_IMG_END_NAMESPACE

#endif