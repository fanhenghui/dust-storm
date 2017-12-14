#include "mi_be_cmd_handler_db_send_ai_evaluation.h"

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_operation_interface.h"

#include "io/mi_image_data.h"
#include "io/mi_configure.h"
#include "io/mi_message.pb.h"

#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mask_label_store.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_logger.h"
#include "mi_model_crosshair.h"
#include "mi_model_dbs_status.h"
#include "mi_model_annotation.h"
#include "mi_app_common_define.h"
#include "mi_app_common_util.h"
#include "mi_app_common_define.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
class OpReceiveAnnotation : public IOperation {
public:
    OpReceiveAnnotation() {}
    virtual ~OpReceiveAnnotation() {}

    virtual int execute() {
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

        //load all(with max limit), and set invisible to result below default threshold
        std::vector<std::string> processing_cache_add;
        const int evaluation_limit = Configure::instance()->get_evaluation_limit();
        if (msg.annotation_size() > evaluation_limit) {
            MI_APPCOMMON_LOG(MI_WARNING) << "more than " << evaluation_limit << " evaluation/annotation result " 
            << msg.annotation_size() << ", clamp it.";
        }
        const int anno_size = (std::min)(evaluation_limit, msg.annotation_size());
        for (int i = 0; i < anno_size; ++i) {
            const MsgAnnotationUnitDB& anno = msg.annotation(i);
            VOISphere voi(Point3(anno.x(),anno.y(),anno.z()), anno.r());
            voi.probability = anno.p();
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

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<OpReceiveAnnotation>(new OpReceiveAnnotation());
    }
};
}

BECmdHandlerDBSendAIEvaluation::BECmdHandlerDBSendAIEvaluation(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

BECmdHandlerDBSendAIEvaluation::~BECmdHandlerDBSendAIEvaluation() {

}

int BECmdHandlerDBSendAIEvaluation::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN recveive DB server DICOM series cmd handler.";
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelDBSStatus> model_dbs_status = AppCommonUtil::get_model_dbs_status(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_dbs_status);

    std::shared_ptr<ModelAnnotation> model_annotation = AppCommonUtil::get_model_annotation(controller);
    if (nullptr == model_annotation) {
        model_dbs_status->push_error_info("model annotation is null.");
    }

    MsgAnnotationCollectionDB msgAnnos;
    if (!msgAnnos.ParseFromArray(buffer,ipcheader.data_len)) {
        model_dbs_status->push_error_info("parse recv dbs AI annotation message failed.");
        return -1;
    }

    if (!model_dbs_status->has_init()) {
        MemShield shield(buffer);//delete buffer later
        MI_APPCOMMON_LOG(MI_INFO) << "receive AI annotation from DBS immediately.";
        //In init
        model_dbs_status->set_ai_annotation();
    
        std::vector<std::string> ids;
        for (int i = 0; i < msgAnnos.annotation_size(); ++i) {
            const MsgAnnotationUnitDB& anno = msgAnnos.annotation(i);
            VOISphere voi(Point3(anno.x(),anno.y(),anno.z()), anno.r());
            voi.probability = anno.p();

            std::stringstream ss;
            ss << clock() << '|' << i; 
            const std::string id = ss.str();
            ids.push_back(id);

            MI_APPCOMMON_LOG(MI_INFO) << "anno item: (" << anno.x() << "," << anno.y() << "," << anno.z() << ") " 
                << anno.r() << ", " << anno.p();

            unsigned char new_label = MaskLabelStore::instance()->acquire_label();
            model_annotation->add_annotation(voi, id, new_label);
        }
        if (!ids.empty()) {
            model_annotation->set_processing_cache(ids);
        } else {
            //TOOD send none annotation message
            model_annotation->set_processing_cache(std::vector<std::string>());
        }

    } else {
        MI_APPCOMMON_LOG(MI_INFO) << "receive delay AI annotation from DBS.";
        //After init
        if(!model_dbs_status->has_query_ai_annotation()) {
            //canceled by user do nothing
            MI_APPCOMMON_LOG(MI_INFO) << "AI annotation query has been canceled by user.";
        } else {
            //create operation and push to BE operation queue
            std::shared_ptr<IOperation> op(new OpReceiveAnnotation());
            op->set_data(ipcheader , buffer);
            op->set_controller(controller);
            controller->get_thread_model()->push_operation_fe(op);
        }
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT recveive DB server DICOM series cmd handler.";
    return 0;
}

MED_IMG_END_NAMESPACE
