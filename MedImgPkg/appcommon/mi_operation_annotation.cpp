#include "mi_operation_annotation.h"

#include "util/mi_ipc_client_proxy.h"
#include "arithmetic/mi_circle.h"

#include "renderalgo/mi_mask_label_store.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_annotation_calculator.h"
#include "renderalgo/mi_volume_infos.h"

#include "mi_model_annotation.h"
#include "mi_app_cell.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_logger.h"
#include "mi_app_common_define.h"
#include "mi_message.pb.h"

MED_IMG_BEGIN_NAMESPACE

OpAnnotation::OpAnnotation() {}

OpAnnotation::~OpAnnotation() {}

int OpAnnotation::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN OpAnnotation.";
    if (_buffer == nullptr || _header._data_len < 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "incompleted annotation message.";
        return -1;
    }

    MsgAnnotationUnit msgAnnotation;
    if (!msgAnnotation.ParseFromArray(_buffer, _header._data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse annotation message failed.";
        return -1;
    }
    const int anno_type = msgAnnotation.type();
    const int anno_id = msgAnnotation.id();
    const int anno_status = msgAnnotation.status();
    const float anno_para0 = msgAnnotation.para0();
    const float anno_para1 = msgAnnotation.para1();
    const float anno_para2 = msgAnnotation.para2(); 

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
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene);
    if (!mpr_scene) {
        MI_APPCOMMON_LOG(MI_ERROR) << "invalid scene when annotation.";
        return -1;
    }

    std::shared_ptr<IModel> model_i = controller->get_model(MODEL_ID_ANNOTATION);
    if (!model_i) {
        MI_APPCOMMON_LOG(MI_ERROR) << "annotation null.";
        return -1;
    }
    std::shared_ptr<ModelAnnotation> model = std::dynamic_pointer_cast<ModelAnnotation>(model_i);
    if (!model) {
        MI_APPCOMMON_LOG(MI_ERROR) << "error model id to acquire annotation model.";
        return -1;
    }

    if (ModelAnnotation::ADD == anno_status) {
        Point2 center_dc(anno_para0, anno_para1); 
        Point3 center_patient;
        if (mpr_scene->get_patient_position(center_dc, center_patient)) {
            VOISphere new_voi(center_patient, FLOAT_EPSILON);
            unsigned char new_label = MaskLabelStore::instance()->acquire_label();
            model->add_annotation(new_voi, new_label);
            model->notify(ModelAnnotation::ADD);
        } else {
            MI_APPCOMMON_LOG(MI_WARNING) << "annotation outside the image.";
            return 0;
        }
    } else if (ModelAnnotation::DELETE == anno_status) {
        const unsigned char label_deleted = model->get_label(anno_id);
        model->remove_annotation(anno_id);
        MaskLabelStore::instance()->recycle_label(label_deleted);
        model->notify(ModelAnnotation::DELETE);

    } else if (ModelAnnotation::MODIFYING == anno_status) {
        //TODO change annotation non-image direction to prevent update non-image when rendering
        Point2 center_dc(anno_para0, anno_para1); 
        Circle circle(Point2(anno_para0, anno_para1), anno_para2);
        VOISphere pre_voi = model->get_annotation(anno_id);
        VOISphere voi = pre_voi;
        if( AnnotationCalculator::dc_circle_update_to_patient_sphere(circle, camera_cal, mpr_scene, voi) ) {
            if(pre_voi != voi) {
                model->modify_center(anno_id, voi.center);
                model->modify_diameter(anno_id, voi.diameter);
                model->notify(ModelAnnotation::MODIFYING);
            } else {
                MI_APPCOMMON_LOG(MI_WARNING) << "annotation circle update to patient sphere does not change.";
                return 0;
            }
        } else {
            MI_APPCOMMON_LOG(MI_WARNING) << "annotation circle update to patient sphere failed.";
            return 0;
        }
    } else if (ModelAnnotation::MODIFY_COMPLETED == anno_status) {
        model->notify(ModelAnnotation::MODIFY_COMPLETED);
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT OpAnnotation.";
    return 0;
}

MED_IMG_END_NAMESPACE