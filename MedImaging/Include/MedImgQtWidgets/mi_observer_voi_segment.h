#ifndef MED_IMAGING_OBSERVER_VOI_SEGMENT_H
#define MED_IMAGING_OBSERVER_VOI_SEGMENT_H

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include "MedImgCommon/mi_observer_interface.h"
#include "MedImgIO/mi_voi.h"
#include "MedImgArithmetic/mi_aabb.h"
#include "MedImgArithmetic/mi_ellipsoid.h"


MED_IMAGING_BEGIN_NAMESPACE

class VOIModel;
class VolumeInfos;
class MPRScene;

class QtWidgets_Export VOISegmentObserver : public IObserver
{
public:
    VOISegmentObserver();
    virtual ~VOISegmentObserver();

    void set_model(std::shared_ptr<VOIModel> model);

    void set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos);

    void set_scenes(std::vector<std::shared_ptr<MPRScene>> scenes );

    virtual void update(int code_id = 0);

private:
    Ellipsoid voi_patient_to_volume(const VOISphere& voi);

    AABBUI get_aabb_i(const Ellipsoid& ellipsoid);

    void segment_i(const Ellipsoid& ellipsoid, const AABBUI& aabb , unsigned char label);

    void recover_i(const AABBUI& aabb , unsigned char label);

private:
    std::weak_ptr<VOIModel> _model;
    std::shared_ptr<VolumeInfos> _volume_infos;
    std::vector<std::shared_ptr<MPRScene>> _scenes;

    //previous status
    std::vector<VOISphere> _pre_vois;
    std::map<unsigned char , AABBUI> _pre_voi_aabbs;
};

MED_IMAGING_END_NAMESPACE


#endif