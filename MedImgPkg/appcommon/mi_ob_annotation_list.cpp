#include "mi_ob_annotation_list.h"

#include "util/mi_ipc_client_proxy.h"

#include "mi_model_annotation.h"
#include "mi_app_common_logger.h"
#include "mi_app_controller.h"
#include "mi_app_common_define.h"
#include "mi_message.pb.h"
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
    MI_APPCOMMON_LOG(MI_DEBUG) << "IN OBAnnotationList";
    std::shared_ptr<ModelAnnotation> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);
    std::string processing_id;
    model->get_processing_cache(processing_id);

    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    //send annotation list changed message(3D)
    IPCDataHeader header;
    header._sender = static_cast<unsigned int>(controller->get_local_pid());
    header._receiver = static_cast<unsigned int>(controller->get_server_pid());
    header._msg_id = COMMAND_ID_BE_SEND_ANNOTATION_LIST;
    header._msg_info0 = 0;
    header._msg_info1 = 0;
    header._data_type = 1;
    header._big_end = 0;

    MsgAnnotationList msg;
    
    switch (code_id) {
        case ModelAnnotation::ADD: {
            const std::string id = model->get_last_annotation();
            if (id != processing_id) {
                MI_APPCOMMON_LOG(MI_ERROR) << "invalid added annotation id.";
                return;
            }
            const int row = model->get_annotation_row(id);
            const VOISphere voi = model->get_annotation(id);
            MsgAnnotationListItem* item = msg.add_item();
            item->set_id(id);
            item->set_info(voi.name);
            item->set_row(row);
            item->set_status(ModelAnnotation::ADD);
            item->set_para0(voi.center.x);
            item->set_para1(voi.center.y);
            item->set_para2(voi.center.z);
            item->set_para3(voi.diameter);

            _pre_vois.insert(std::make_pair(id, OBAnnotationList::VOIUnit(voi,row)));
            
            break;
        }
        case ModelAnnotation::DELETE: {
            //get deleted voi
            for (auto it = _pre_vois.begin(); it != _pre_vois.end(); ++it) {
                if(it->first == processing_id) {
                    MsgAnnotationListItem* item = msg.add_item();
                    const VOISphere& voi = it->second.voi;
                    const int row = it->second.row;
                    item->set_id(processing_id);
                    item->set_info(voi.name);
                    item->set_row(row);
                    item->set_status(ModelAnnotation::DELETE);
                    item->set_para0(voi.center.x);
                    item->set_para1(voi.center.y);
                    item->set_para2(voi.center.z);
                    item->set_para3(voi.diameter);

                    it = _pre_vois.erase(it);
                    break;
                }
            }
            break;
        }
        case ModelAnnotation::MODIFYING: {
            //get deleted voi
            for (auto it = _pre_vois.begin(); it != _pre_vois.end(); ++it) {
                if(it->first == processing_id) {
                    MsgAnnotationListItem* item = msg.add_item();
                    const VOISphere& voi = it->second.voi;
                    const int row = it->second.row;
                    item->set_id(processing_id);
                    item->set_info(voi.name);
                    item->set_row(row);
                    item->set_status(ModelAnnotation::MODIFYING);
                    item->set_para0(voi.center.x);
                    item->set_para1(voi.center.y);
                    item->set_para2(voi.center.z);
                    item->set_para3(voi.diameter);

                    it->second.voi = model->get_annotation(processing_id);
                    break;
                }
            }
            break;
        }
        default:
            return;
    }

    const int buffer_size = msg.ByteSize();
    if (buffer_size <= 0) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialized annotation list msg buffer length less than 0.";
        return;
    }
    char* buffer = new char[buffer_size];
    if (!msg.SerializeToArray(buffer, buffer_size)) {
        MI_APPCOMMON_LOG(MI_ERROR) << "serialized annotation list msg buffer failed.";
        delete [] buffer;
        buffer = nullptr;
        return;
    }

    header._data_len = static_cast<unsigned int>(buffer_size);
    controller->get_client_proxy()->async_send_message(header, buffer);

    MI_APPCOMMON_LOG(MI_DEBUG) << "OUT OBAnnotationList";
}

MED_IMG_END_NAMESPACE