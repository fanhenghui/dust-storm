#ifndef MED_IMG_REVIEW_MI_OB_ANNOTATION_SEGMENTATION_H
#define MED_IMG_REVIEW_MI_OB_ANNOTATION_SEGMENTATION_H

#include <vector>
#include <map>
#include <memory>
#include "mi_review_common.h"
#include "util/mi_observer_interface.h"
#include "arithmetic/mi_ellipsoid.h"
#include "arithmetic/mi_aabb.h"
#include "io/mi_voi.h"

MED_IMG_BEGIN_NAMESPACE

class ModelAnnotation;
class VolumeInfos;
class MPRScene;
class VRScene;
class OBAnnotationSegment : public IObserver {
public:
    OBAnnotationSegment();
    virtual ~OBAnnotationSegment();

    void set_model(std::shared_ptr<ModelAnnotation> model);
    void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);
    void set_mpr_scenes(std::vector<std::shared_ptr<MPRScene>> scenes);
    void set_vr_scenes(std::vector<std::shared_ptr<VRScene>> scenes);

    virtual void update(int code_id = 0);

private:
    Ellipsoid voi_patient_to_volume(const VOISphere& voi);
    int get_aabb_i(const Ellipsoid& ellipsoid, AABBUI& aabb);
    void segment_i(const Ellipsoid& ellipsoid, const AABBUI& aabb , unsigned char label);
    void recover_i(const AABBUI& aabb , unsigned char label);

    void update_aabb_i(const AABBUI& aabb);

private:
    std::weak_ptr<ModelAnnotation> _model;
    std::shared_ptr<VolumeInfos> _volume_infos;
    std::vector<std::shared_ptr<MPRScene>> _mpr_scenes;
    std::vector<std::shared_ptr<VRScene>> _vr_scenes;

    //previous status
    std::vector<VOISphere> _pre_vois;
    std::map<unsigned char , AABBUI> _pre_voi_aabbs;
};

MED_IMG_END_NAMESPACE
#endif
