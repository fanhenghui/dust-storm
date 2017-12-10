#include "mi_be_operation_fe_locate.h"

#include "util/mi_ipc_client_proxy.h"

#include "arithmetic/mi_point2.h"
#include "arithmetic/mi_point3.h"
#include "arithmetic/mi_vector3.h"
#include "arithmetic/mi_arithmetic_utils.h"

#include "mi_message.pb.h"
#include "glresource/mi_gl_context.h"

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
#include "mi_app_thread_model.h"
#include "mi_app_common_define.h"

MED_IMG_BEGIN_NAMESPACE

BEOpFELocate::BEOpFELocate() {}

BEOpFELocate::~BEOpFELocate() {}

int BEOpFELocate::execute() {
    if (_buffer == nullptr || _header.data_len < 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "incompleted locate message.";
        return -1;
    }

    MsgCrosshair msgCrosshair;
    if (!msgCrosshair.ParseFromArray(_buffer, _header.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse crosshair message failed.";
        return -1;
    }

    const float cx = msgCrosshair.cx();
    const float cy = msgCrosshair.cy();
    //const float l0_a = msgCrosshair.l0_a();
    //const float l0_b = msgCrosshair.l0_b();
    //const float l1_a = msgCrosshair.l1_a();
    //const float l1_b = msgCrosshair.l1_b();
    msgCrosshair.Clear();
    const Point2 pt_cross(cx, cy);

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    const unsigned int cell_id = _header.cell_id;
    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    APPCOMMON_CHECK_NULL_EXCEPTION(cell);
    std::shared_ptr<SceneBase> scene = cell->get_scene();
    APPCOMMON_CHECK_NULL_EXCEPTION(scene);

    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
    if (mpr_scene) {
        return mpr_locate_i(cell, mpr_scene, pt_cross);
    }

    std::shared_ptr<VRScene> vr_scene = std::dynamic_pointer_cast<VRScene>(scene);
    if (vr_scene) {
        return vr_locate_i(cell, vr_scene, pt_cross);
    }

    MI_APPCOMMON_LOG(MI_ERROR) << "invalid scene type when locate.";
    return -1;
}

int BEOpFELocate::mpr_locate_i(std::shared_ptr<AppCell> cell, std::shared_ptr<MPRScene> mpr_scene, const Point2& pt_cross) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFELocate MPR.";

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_CROSSHAIR);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_);
    std::shared_ptr<ModelCrosshair> model = std::dynamic_pointer_cast<ModelCrosshair>(model_);
    APPCOMMON_CHECK_NULL_EXCEPTION(model);

    model->locate(mpr_scene, pt_cross);

    //TODO adjust FE position when locate outside volume range
    int width(-1),height(-1);
    mpr_scene->get_display_size(width, height);

    //update located mpr cell's crosshair none-image to prevent op loop
    std::shared_ptr<IAppNoneImage> app_none_image_ = cell->get_none_image();
    APPCOMMON_CHECK_NULL_EXCEPTION(app_none_image_);
    std::shared_ptr<AppNoneImage> app_none_image = std::dynamic_pointer_cast<AppNoneImage>(app_none_image_);
    APPCOMMON_CHECK_NULL_EXCEPTION(app_none_image);
    std::shared_ptr<INoneImg> crosshair_nonimg_ = app_none_image->get_none_image_item(Crosshair);
    APPCOMMON_CHECK_NULL_EXCEPTION(crosshair_nonimg_);
    std::shared_ptr<NoneImgCrosshair> crosshair_nonimg = std::dynamic_pointer_cast<NoneImgCrosshair>(crosshair_nonimg_);
    APPCOMMON_CHECK_NULL_EXCEPTION(crosshair_nonimg);

    crosshair_nonimg->check_dirty();
    crosshair_nonimg->update();

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFELocate MPR.";
    return 0;
}

int BEOpFELocate::vr_locate_i(std::shared_ptr<AppCell> cell, std::shared_ptr<VRScene> vr_scene, const Point2& pt_cross) {

    MI_APPCOMMON_LOG(MI_TRACE) << "IN BEOpFELocate VR.";

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<AppThreadModel> thread_model = controller->get_thread_model();
    APPCOMMON_CHECK_NULL_EXCEPTION(thread_model);
    std::shared_ptr<GLContext> gl_context = thread_model->get_gl_context();
    APPCOMMON_CHECK_NULL_EXCEPTION(gl_context);
    std::shared_ptr<IModel> model_ = controller->get_model(MODEL_ID_CROSSHAIR);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_);
    std::shared_ptr<ModelCrosshair> model = std::dynamic_pointer_cast<ModelCrosshair>(model_);
    APPCOMMON_CHECK_NULL_EXCEPTION(model);

    //cache ray end 
    gl_context->make_current(OPERATION_CONTEXT);
    vr_scene->cache_ray_end();
    gl_context->make_noncurrent();

    //calculate cross world position
    Point3 pt_cross_w;
    const bool got_it = vr_scene->get_ray_end(pt_cross, pt_cross_w);

    if (got_it) {
        model->locate(pt_cross_w);
        //update located mpr cell's crosshair none-image to prevent op loop
        std::shared_ptr<IAppNoneImage> app_none_image_ = cell->get_none_image();
        APPCOMMON_CHECK_NULL_EXCEPTION(app_none_image_);
        std::shared_ptr<AppNoneImage> app_none_image = std::dynamic_pointer_cast<AppNoneImage>(app_none_image_);
        APPCOMMON_CHECK_NULL_EXCEPTION(app_none_image);
        std::shared_ptr<INoneImg> crosshair_nonimg_ = app_none_image->get_none_image_item(Crosshair);
        APPCOMMON_CHECK_NULL_EXCEPTION(crosshair_nonimg_);
        std::shared_ptr<NoneImgCrosshair> crosshair_nonimg = std::dynamic_pointer_cast<NoneImgCrosshair>(crosshair_nonimg_);
        APPCOMMON_CHECK_NULL_EXCEPTION(crosshair_nonimg);

        crosshair_nonimg->check_dirty();
        crosshair_nonimg->update();
    } else {
        //TODO adjust FE position when locate outside volume range
        int width(-1),height(-1);
        vr_scene->get_display_size(width, height);
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BEOpFELocate VR.";
    return 0;
}

MED_IMG_END_NAMESPACE