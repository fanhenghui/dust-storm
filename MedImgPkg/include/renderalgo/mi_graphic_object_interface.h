#ifndef MEDIMGRENDERALGO_MI_GRIPHIC_OBJECT_INTERFACE_H
#define MEDIMGRENDERALGO_MI_GRIPHIC_OBJECT_INTERFACE_H

#include "renderalgo/mi_render_algo_export.h"
#include "arithmetic/mi_camera_base.h"
#include "renderalgo/mi_volume_infos.h"

MED_IMG_BEGIN_NAMESPACE

class IGraphicObject 
{
public:
    IGraphicObject() {};
    virtual ~IGraphicObject() {};

    void set_camera(std::shared_ptr<CameraBase> camera) {
        _camera = camera;
    }

    void set_volume_info(std::shared_ptr<VolumeInfos> volume_infos) {
        _volume_infos = volume_infos;
    }

    virtual void initialize() {}

    virtual void render(int code = 0) = 0;

protected:
    std::shared_ptr<CameraBase> _camera;
    std::shared_ptr<VolumeInfos> _volume_infos;
};

MED_IMG_END_NAMESPACE

#endif