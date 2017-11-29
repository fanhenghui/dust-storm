#include "mi_cmd_handler_recv_dbs_ai_annotation.h"

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_client_proxy.h"

#include "io/mi_image_data.h"

#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mask_label_store.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_logger.h"
#include "mi_message.pb.h"
#include "mi_model_dbs_status.h"
#include "mi_model_annotation.h"
#include "mi_app_common_define.h"
#include "mi_app_common_util.h"
#include "mi_app_config.h"
#include "mi_app_common_define.h"
#include "mi_operation_receive_annotation.h"


MED_IMG_BEGIN_NAMESPACE

CmdHandlerRecvDBSAIAnno::CmdHandlerRecvDBSAIAnno(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

CmdHandlerRecvDBSAIAnno::~CmdHandlerRecvDBSAIAnno() {

}

int CmdHandlerRecvDBSAIAnno::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
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
        const float possibility_threshold = AppConfig::instance()->get_nodule_possibility_threshold();
        for (int i = 0; i < msgAnnos.annotation_size(); ++i) {
            const MsgAnnotationUnitDB& anno = msgAnnos.annotation(i);
            if (anno.p() < possibility_threshold) {
                continue;
            }
            VOISphere voi(Point3(anno.x(),anno.y(),anno.z()), anno.r());
            voi.para0 = anno.p();

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
            controller->get_thread_model()->push_operation(op);
        }
    }

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT recveive DB server DICOM series cmd handler.";
    return 0;
}

MED_IMG_END_NAMESPACE
