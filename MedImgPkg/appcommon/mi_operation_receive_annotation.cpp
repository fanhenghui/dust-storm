#include "mi_operation_receive_annotation.h"

#include "util/mi_ipc_client_proxy.h"
#include "arithmetic/mi_circle.h"

#include "renderalgo/mi_mask_label_store.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_vr_scene.h"
#include "renderalgo/mi_annotation_calculator.h"
#include "renderalgo/mi_volume_infos.h"

#include "mi_model_annotation.h"
#include "mi_model_crosshair.h"
#include "mi_model_dbs_status.h"
#include "mi_app_cell.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_none_image_item.h"
#include "mi_app_none_image.h"
#include "mi_message.pb.h"
#include "mi_app_common_util.h"
#include "mi_app_config.h"

MED_IMG_BEGIN_NAMESPACE

OpReceiveAnnotation::OpReceiveAnnotation() {}

OpReceiveAnnotation::~OpReceiveAnnotation() {}

int OpReceiveAnnotation::execute() {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN OpReceiveAnnotation.";
    if (_buffer == nullptr || _header.data_len < 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "incompleted annotation message from DBS.";
        return -1;
    }

    MsgAnnotationCollectionDB msg;
    if (!msg.ParseFromArray(_buffer, _header.data_len)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "parse annotation message from DBS failed.";
        return -1;
    }

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<ModelAnnotation> model_annotation = AppCommonUtil::get_model_annotation(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_annotation);
    std::shared_ptr<ModelDBSStatus> model_dbs_status = AppCommonUtil::get_model_dbs_status(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_dbs_status);

    //Delete old nodules if has
    std::map<std::string, ModelAnnotation::AnnotationUnit> old_annos = model_annotation->get_annotations();
    if (!old_annos.empty()) {
        std::vector<std::string> processing_cache_del;
        for (auto it = old_annos.begin(); it != old_annos.end(); ++it) {
            const std::string& id = it->first;
            processing_cache_del.push_back(id);
            const unsigned char label_del = model_annotation->get_label(id);
            model_annotation->remove_annotation(id);
            MaskLabelStore::instance()->recycle_label(label_del);
        }
        model_annotation->set_processing_cache(processing_cache_del);
        model_annotation->notify(ModelAnnotation::DELETE);
    }

    model_dbs_status->set_ai_annotation();//AI annotation flag

    std::vector<std::string> processing_cache_add;
    const float possibility_threshold = AppConfig::instance()->get_nodule_possibility_threshold();
    for (int i = 0; i < msg.annotation_size(); ++i) {
        const MsgAnnotationUnitDB& anno = msg.annotation(i);
        if (anno.p() < possibility_threshold) {
            continue;
        }
        VOISphere voi(Point3(anno.x(),anno.y(),anno.z()), anno.r());
        voi.para0 = anno.p();
        std::stringstream ss;
        ss << clock() << '|' << i; 
        const std::string id = ss.str();
        processing_cache_add.push_back(id);
        MI_APPCOMMON_LOG(MI_INFO) << "anno item: (" << anno.x() << "," << anno.y() << "," << anno.z() << ") " 
            << anno.r() << ", " << anno.p();
        unsigned char new_label = MaskLabelStore::instance()->acquire_label();
        model_annotation->add_annotation(voi, id, new_label);
    }
    if (!processing_cache_add.empty()) {
        model_annotation->set_processing_cache(processing_cache_add);
        model_annotation->notify(ModelAnnotation::ADD);
    } else {
        //TOOD send none annotation message
        model_annotation->set_processing_cache(std::vector<std::string>());
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT OpReceiveAnnotation.";
    return 0;
}

MED_IMG_END_NAMESPACE