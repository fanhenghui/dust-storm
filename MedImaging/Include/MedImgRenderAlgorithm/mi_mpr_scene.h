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

    void set_mask_overlay_mode(MaskOverlayMode mode);

    void set_mask_overlay_color(std::map<unsigned char , RGBAUnit> colors);
    void set_mask_overlay_color(RGBAUnit color , unsigned char label);

    virtual void rotate(const Point2& pre_pt , const Point2& cur_pt);

    virtual void zoom(const Point2& pre_pt , const Point2& cur_pt);

    virtual void pan(const Point2& pre_pt , const Point2& cur_pt);

    //Default positive direction
    // Axial F->H
    // Sagittal R->L
    // Coronal A -> P
    void page(int step);

    void page_to(int page);

    //Call to initialize MPR placement
    void place_mpr(ScanSliceType type);

    bool get_volume_position(const Point2& pt_dc, Point3& pos_v);

    bool get_world_position(const Point2& pt_dc, Point3& pos_w);

    bool get_patient_position(const Point2& pt_dc, Point3& pos_p);

    Plane to_plane()const;

protected:
private:
};

MED_IMAGING_END_NAMESPACE

#endif