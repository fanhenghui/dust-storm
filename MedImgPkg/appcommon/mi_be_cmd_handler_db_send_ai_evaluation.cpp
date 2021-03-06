#include "mi_be_cmd_handler_db_send_ai_evaluation.h"

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_operation_interface.h"

#include "io/mi_image_data.h"
#include "io/mi_configure.h"
#include "io/mi_protobuf.h"

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
        APPCOMMON_CHECK_NULL_EXCEPTION(_buffer);

        MsgAnnotationCollectionDB msg;
        if (0 != protobuf_parse(_buffer, _header.data_len, msg)) {
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

        //add all(with count limit) to model ,but changed part with high probability
        const float probability = model_annotation->get_probability_threshold();
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
            ss << "eva-" << clock() << '-' << i; 
            const std::string id = ss.str();
            
            if (voi.probability >= probability) {
                processing_cache_add.push_back(id);
            }

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

    CREATE_EXTENDS_OP(OpReceiveAnnotation)
};
}

BECmdHandlerDBSendAIEvaluation::BECmdHandlerDBSendAIEvaluation(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

BECmdHandlerDBSendAIEvaluation::~BECmdHandlerDBSendAIEvaluation() {

}

int BECmdHandlerDBSendAIEvaluation::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerDBSendAIEvaluation.";
    
    APPCOMMON_CHECK_NULL_EXCEPTION(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<ModelDBSStatus> model_dbs_status = AppCommonUtil::get_model_dbs_status(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_dbs_status);

    std::shared_ptr<ModelAnnotation> model_annotation = AppCommonUtil::get_model_annotation(controller);
    if (nullptr == model_annotation) {
        model_dbs_status->push_error_info("model annotation is null.");
    }

    MsgAnnotationCollectionDB msg;
    if (0 != protobuf_parse(buffer, ipcheader.data_len, msg)) {
        model_dbs_status->push_error_info("parse recv dbs AI annotation message failed.");
        return -1;
    }

    if (!model_dbs_status->has_init()) {
        MemShield shield(buffer);//delete buffer later
        MI_APPCOMMON_LOG(MI_INFO) << "receive AI annotation from DBS immediately.";
        //In init
        model_dbs_status->set_ai_annotation();
    
        //add all(with count limit) to model ,but changed part with high probability
        const float probability = model_annotation->get_probability_threshold();
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
            ss << "eva-" << clock() << '-' << i; 
            const std::string id = ss.str();

            if (voi.probability >= probability) {
                processing_cache_add.push_back(id);
            }
            
            MI_APPCOMMON_LOG(MI_INFO) << "anno item: (" << anno.x() << "," << anno.y() << "," << anno.z() << ") " 
                << anno.r() << ", " << anno.p();

            unsigned char new_label = MaskLabelStore::instance()->acquire_label();
            model_annotation->add_annotation(voi, id, new_label);
        }
        if (!processing_cache_add.empty()) {
            model_annotation->set_processing_cache(processing_cache_add);
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
            //TOOD check current series uid equal to received one 

            //create operation and push to BE operation queue
            std::shared_ptr<IOperation> op(new OpReceiveAnnotation());
            op->set_data(ipcheader , buffer);
            op->set_controller(controller);
            controller->get_thread_model()->push_operation_fe(op);
        }
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerDBSendAIEvaluation.";
    return 0;
}

MED_IMG_END_NAMESPACE
