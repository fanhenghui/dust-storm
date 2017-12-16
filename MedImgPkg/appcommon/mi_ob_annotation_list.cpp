#include "mi_ob_annotation_list.h"

#include "util/mi_ipc_client_proxy.h"
#include "io/mi_protobuf.h"

#include "mi_model_annotation.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

OBAnnotationList::OBAnnotationList() {

}

OBAnnotationList::~OBAnnotationList() {

}

void OBAnnotationList::set_model(std::shared_ptr<ModelAnnotation> model) {
    _model = model;
}

void OBAnnotationList::set_controller(std::shared_ptr<AppController> controller) {
    _controller = controller;
}

void OBAnnotationList::update(int code_id) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN OBAnnotationList";
    std::shared_ptr<ModelAnnotation> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);
    std::vector<std::string> processing_ids;
    model->get_processing_cache(processing_ids);

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //send annotation list changed message(3D)
    IPCDataHeader header;
    header.sender = static_cast<unsigned int>(controller->get_local_pid());
    header.receiver = static_cast<unsigned int>(controller->get_server_pid());
    header.msg_id = COMMAND_ID_FE_BE_SEND_ANNOTATION_LIST;

    const float probability = model->get_probability_threshold();    
    MsgNoneImgAnnotations msg;
    switch (code_id) {
        case ModelAnnotation::ADD: {
            for (auto it = processing_ids.begin(); it != processing_ids.end(); ++it) {
                const std::string id = *it;
                const VOISphere voi = model->get_annotation(id);
                if (voi.probability < probability) {
                    continue;
                }
                MsgAnnotationUnit* item = msg.add_annotation();
                item->set_id(id);
                item->set_info(voi.name);
                item->set_status(ModelAnnotation::ADD);
                item->set_para0(voi.center.x);
                item->set_para1(voi.center.y);
                item->set_para2(voi.center.z);
                item->set_para3(voi.diameter);
                item->set_probability(voi.probability);
                _pre_vois.insert(std::make_pair(id, voi));
            }          
            break;
        }
        case ModelAnnotation::DELETE: {
            //get deleted voi
            for (auto it = _pre_vois.begin(); it != _pre_vois.end(); ) {
                bool got_deleted = false;
                for (auto it2 = processing_ids.begin(); it2 != processing_ids.end(); ++it2) {
                    const std::string id = *it2;
                    if(it->first == id) {
                        MsgAnnotationUnit* item = msg.add_annotation();
                        const VOISphere& voi = it->second;
                        item->set_id(id);
                        item->set_info(voi.name);
                        item->set_status(ModelAnnotation::DELETE);
                        item->set_para0(voi.center.x);
                        item->set_para1(voi.center.y);
                        item->set_para2(voi.center.z);
                        item->set_para3(voi.diameter);
                        item->set_probability(voi.probability);
    
                        it = _pre_vois.erase(it);
                        got_deleted = true;
                        break;
                    }
                }
                if (!got_deleted) {
                    ++it;
                }
            }
            break;
        }
        case ModelAnnotation::MODIFYING: {
            //get deleted voi
            for (auto it = _pre_vois.begin(); it != _pre_vois.end(); ++it) {
                for (auto it2 = processing_ids.begin(); it2 != processing_ids.end(); ++it2) {
                    const std::string id = *it2;
                    if(it->first == id) {
                        MsgAnnotationUnit* item = msg.add_annotation();
                        const VOISphere& voi = it->second;
                        item->set_id(id);
                        item->set_info(voi.name);
                        item->set_status(ModelAnnotation::MODIFYING);
                        item->set_para0(voi.center.x);
                        item->set_para1(voi.center.y);
                        item->set_para2(voi.center.z);
                        item->set_para3(voi.diameter);
                        item->set_probability(voi.probability);
    
                        it->second = model->get_annotation(id);
                        break;
                    }
                }
            }
            break;
        }
        default:
            return;
    }

    char* buffer = nullptr;
    int buffer_size = 0;
    if (0 != protobuf_serialize(msg, buffer, buffer_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialized annotation list msg buffer failed.";
        return;
    }
    msg.Clear();
    
    header.data_len = static_cast<unsigned int>(buffer_size);
    controller->get_client_proxy()->sync_send_data(header, buffer);
    if (buffer != nullptr) {
        delete [] buffer;
    }
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT OBAnnotationList";
}

MED_IMG_END_NAMESPACE