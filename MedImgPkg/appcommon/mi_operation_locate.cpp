#include "mi_operation_locate.h"

#include "util/mi_ipc_client_proxy.h"

#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_vector3.h"
#include "arithmetic/mi_arithmetic_utils.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_annotation_calculator.h"
#include "renderalgo/mi_volume_infos.h"

#include "mi_model_crosshair.h"
#include "mi_app_cell.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_none_image_item.h"
#include "mi_app_none_image.h"
#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

OpLocate::OpLocate() {}

OpLocate::~OpLocate() {}

int OpLocate::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN OpLocate.";
    if (_buffer == nullptr || _header._data_len < 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "incompleted locate message.";
        return -1;
    }

    MsgCrosshair msgCrosshair;
    if (!msgCrosshair.ParseFromArray(_buffer, _header._data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse crosshair message failed.";
        return -1;
    }

    const float cx = msgCrosshair.cx();
    const float cy = msgCrosshair.cy();
    const float l0_a = msgCrosshair.l0_a();
    const float l0_b = msgCrosshair.l0_b();
    const float l1_a = msgCrosshair.l1_a();
    const float l1_b = msgCrosshair.l1_b();


    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<VolumeInfos> volumeinfos = controller->get_volume_infos();
    APPCOMMON_CHECK_NULL_EXCEPTION(volumeinfos);
    std::shared_ptr<CameraCalculator> camera_cal = volumeinfos->get_camera_calculator();
    APPCOMMON_CHECK_NULL_EXCEPTION(camera_cal);

    const unsigned int cell_id = _header._cell_id;
    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    APPCOMMON_CHECK_NULL_EXCEPTION(cell);
    std::shared_ptr<SceneBase> scene = cell->get_scene();
    APPCOMMON_CHECK_NULL_EXCEPTION(scene);
    int width(-1),height(-1);
    scene->get_display_size(width, height);

    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_CROSSHAIR);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_);
    std::shared_ptr<ModelCrosshair> model = std::dynamic_pointer_cast<ModelCrosshair>(model_);
    APPCOMMON_CHECK_NULL_EXCEPTION(model);

    std::shared_ptr<IAppNoneImage> app_none_image_ = cell->get_none_image();
    APPCOMMON_CHECK_NULL_EXCEPTION(app_none_image_);
    std::shared_ptr<AppNoneImage> app_none_image = std::dynamic_pointer_cast<AppNoneImage>(app_none_image_);
    APPCOMMON_CHECK_NULL_EXCEPTION(app_none_image);
    std::shared_ptr<INoneImg> crosshair_nonimg_ = app_none_image->get_none_image_item(Crosshair);
    APPCOMMON_CHECK_NULL_EXCEPTION(crosshair_nonimg_);
    std::shared_ptr<NoneImgCrosshair> crosshair_nonimg = std::dynamic_pointer_cast<NoneImgCrosshair>(crosshair_nonimg_);
    APPCOMMON_CHECK_NULL_EXCEPTION(crosshair_nonimg);

    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
    if (mpr_scene) {
        //TODO just consider orthogonal situation tmp
        //TODO consider border beyond volume boundary
        model->locate(mpr_scene, Point2(cx, cy));
        crosshair_nonimg->check_dirty();
        crosshair_nonimg->update();
    } else {
        std::shared_ptr<VRScene> vr_scene = std::dynamic_pointer_cast<VRScene>(scene);
        if (vr_scene) {

        } else {
            MI_APPCOMMON_LOG(MI_ERROR) << "invalid cell type. not vr/mpr.";
        }
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT OpLocate.";
    return 0;
}


MED_IMG_END_NAMESPACE

