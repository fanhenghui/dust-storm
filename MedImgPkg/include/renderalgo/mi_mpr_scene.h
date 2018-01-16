#ifndef MEDIMGRENDERALGO_MPR_SCENE_H
#define MEDIMGRENDERALGO_MPR_SCENE_H

#include "arithmetic/mi_plane.h"
#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_ray_cast_scene.h"

MED_IMG_BEGIN_NAMESPACE

class RenderAlgo_Export MPRScene : public RayCastScene {
public:
    MPRScene(RayCastingStrategy strategy, GPUPlatform platfrom);
    MPRScene(int width, int height, RayCastingStrategy strategy, GPUPlatform platfrom);
    virtual ~MPRScene();

    virtual void initialize();
    virtual void render_to_back();

    void set_mask_overlay_mode(MaskOverlayMode mode);

    void set_mask_overlay_color(std::map<unsigned char, RGBAUnit> colors);
    void set_mask_overlay_color(RGBAUnit color, unsigned char label);
    void set_mask_overlay_opacity(float opacity);

    virtual void rotate(const Point2& pre_pt, const Point2& cur_pt);

    virtual void zoom(const Point2& pre_pt, const Point2& cur_pt);

    virtual void pan(const Point2& pre_pt, const Point2& cur_pt);

    // Default positive direction
    // Axial F->H
    // Sagittal R->L
    // Coronal A -> P
    void page(int step);

    void page_to(int page);

    // Call to initialize MPR placement
    void place_mpr(ScanSliceType type);

    bool get_volume_position(const Point2& pt_dc, Point3& pos_v);

    bool get_world_position(const Point2& pt_dc, Point3& pos_w);

    bool get_patient_position(const Point2& pt_dc, Point3& pos_p);

    Plane to_plane() const;

protected:
private:
    bool _mpr_init;
};

MED_IMG_END_NAMESPACE

#endif