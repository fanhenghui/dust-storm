#include "mi_ai_operation_evaluate.h"

#include "util/mi_file_util.h"
#include "util/mi_memory_shield.h"

#include "io/mi_nodule_set.h"
#include "io/mi_nodule_set_parser.h"

#include "appcommon/mi_app_db.h"
#include "appcommon/mi_message.pb.h"
#include "appcommon/mi_app_config.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_ai_server_controller.h"
#include "mi_ai_server_thread_model.h"
#include "mi_ai_lung_evaluate_py_wrapper.h"

MED_IMG_BEGIN_NAMESPACE

AIOpEvaluate::AIOpEvaluate() {

}

AIOpEvaluate::~AIOpEvaluate() {

}

char* serialize_msg(MsgEvaluationResponse& msg, int& msg_size) {
    msg_size = msg.ByteSize();
    if (msg_size <= 0) {
        return nullptr;
    }
    char* msg_buffer = new char[msg_size];
    if (!msg.SerializeToArray(msg_buffer,msg_size)) {
        delete [] msg_buffer;
        msg_buffer = nullptr;
        return nullptr; 
    } else {
        return msg_buffer;
    }
}

int notify_dbs(MsgEvaluationResponse& msg, std::shared_ptr<AIServerController> controller) {
    int msg_size = 0;                
    char* msg_buffer = serialize_msg(msg,msg_size);
    if (msg_buffer) {
        IPCDataHeader header;
        header.msg_id = COMMAND_ID_AI_DB_OPERATION;
        header.msg_info1 = OPERATION_ID_DB_RECEIVE_AI_EVALUATION;
        header.data_len = msg_size;
        msg.Clear();
        controller->get_thread_model()->async_send_data(new IPCPackage(header,msg_buffer));
        return 0;
    } else {
        msg.Clear();
        MI_AISERVER_LOG(MI_ERROR) << "notify DBS failed.";
        return -1;
    }
}

int AIOpEvaluate::execute() {
    MI_AISERVER_LOG(MI_TRACE) << "IN Inference operation.";

    AISERVER_CHECK_NULL_EXCEPTION(_buffer);

    std::shared_ptr<AIServerController> controller  = get_controller<AIServerController>();
    AISERVER_CHECK_NULL_EXCEPTION(controller);

    //response message
    MsgEvaluationResponse msg_res;

    //parse request message
    MsgEvaluationRequest msg_req;
    if (!msg_req.ParseFromArray(_buffer, _header.data_len)) {
        MI_AISERVER_LOG(MI_ERROR) << "parse evaluation request message failed.";
        msg_res.set_status(-1);
        msg_res.set_err_msg("parse request message failed.");
        return notify_dbs(msg_res, controller);
    }

    const std::string series_id = msg_req.series_uid();
    const std::string dcm_path = msg_req.dcm_path();
    std::string ai_anno_path = msg_req.ai_anno_path();
    std::string ai_im_data_path = msg_req.ai_im_data_path();
    bool recal_im_data = msg_req.recal_im_data();
    const uint64_t socket_id = msg_req.client_socket_id();
    msg_req.Clear();

    if (ai_im_data_path.empty() || recal_im_data) {
        recal_im_data = true;
        ai_im_data_path = dcm_path + "/" + series_id + ".npy";
    }
    if (ai_anno_path.empty()) {
        ai_anno_path = dcm_path + "/" + series_id + ".csv";
    }

    msg_res.set_series_uid(series_id);
    msg_res.set_ai_anno_path(ai_anno_path);
    msg_res.set_ai_im_data_path(ai_im_data_path);
    msg_res.set_recal_im_data(recal_im_data);
    msg_res.set_client_socket_id(socket_id);//TODO DBS增加 计算状态机以及等待计算结果的observer后可以删除

    //get python running path
    const std::string pytorch_path = AppConfig::instance()->get_pytorch_path();
    const std::string py_interface_path = AppConfig::instance()->get_py_interface_path();

    //debug print info
    // MI_AISERVER_LOG(MI_DEBUG) << "IN AI evaluation operation:";
    // MI_AISERVER_LOG(MI_DEBUG) << "pytorch path: " << pytorch_path;
    // MI_AISERVER_LOG(MI_DEBUG) << "py interface path: " << py_interface_path;
    // MI_AISERVER_LOG(MI_DEBUG) << "series ID: " << series_id;
    // MI_AISERVER_LOG(MI_DEBUG) << "DICOM direction: " << dcm_path;
    // MI_AISERVER_LOG(MI_DEBUG) << "AI intermediate data path: "  << ai_im_data_path;
    // MI_AISERVER_LOG(MI_DEBUG) << "msg buffer length: " << _header.data_len;

    std::shared_ptr<medical_ai::AILungEvaulatePyWrapper> wrapper(new medical_ai::AILungEvaulatePyWrapper());
    if(-1 == wrapper->init(pytorch_path.c_str() ,py_interface_path.c_str()) ){
        MI_AISERVER_LOG(MI_FATAL) << "init python env failed.";
        msg_res.set_status(-1);
        msg_res.set_err_msg("init python env failed.");
        return notify_dbs(msg_res, controller);
    }

    //Preprocess
    if (recal_im_data) {
        MI_AISERVER_LOG(MI_DEBUG) << "AI lung preprocess";
        char* buffer_im_data = nullptr;
        int buffer_im_data_size = 0;
        if (-1 == wrapper->preprocess(dcm_path.c_str(), buffer_im_data, buffer_im_data_size) ){
            const char* err = wrapper->get_last_err();
            MI_AISERVER_LOG(MI_ERROR) << "AI lung preprocess failed :" << err;
            msg_res.set_status(-1);
            msg_res.set_err_msg("AI lung preprocess failed.");
            return notify_dbs(msg_res, controller);
        }

        MemShield shield(buffer_im_data);
        if (0 != FileUtil::write_raw(ai_im_data_path, buffer_im_data, buffer_im_data_size)) {
            MI_AISERVER_LOG(MI_ERROR) << "write AI intermediate data to: " << ai_im_data_path << " failed.";
            msg_res.set_status(-1);
            msg_res.set_err_msg("write AI intermediate data to disk failed.");
            return notify_dbs(msg_res, controller);
        }
    }

    //Evaluate
    MI_AISERVER_LOG(MI_DEBUG) << "AI lung evalulate";
    medical_ai::AILungEvaulatePyWrapper::VPredictedNodules nodules;
    if(-1 == wrapper->evaluate(ai_im_data_path.c_str(), nodules)) {
        const char* err = wrapper->get_last_err();
        MI_AISERVER_LOG(MI_ERROR) << "evalulate series: " << series_id << " failed: " << err;
        msg_res.set_status(-1);
        msg_res.set_err_msg("evalulate series failed.");
        return notify_dbs(msg_res, controller);
    }

    //encode and write to disk
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    for (size_t i=0; i<nodules.size(); ++i) {
        VOISphere voi(Point3(nodules[i].fX,nodules[i].fY,nodules[i].fZ),nodules[i].fRadius*2.0);
        voi.para0 = nodules[i].fProb;
        nodule_set->add_nodule(voi);
    }

    NoduleSetParser ns_parser;
    ns_parser.set_series_id(series_id);
    if(IO_SUCCESS != ns_parser.save_as_csv(ai_anno_path, nodule_set)) {
        MI_AISERVER_LOG(MI_ERROR) << "save evaluated result to " << ai_anno_path << " failed.";
        msg_res.set_status(-1);
        msg_res.set_err_msg("save evaluated result failed.");
        return notify_dbs(msg_res, controller);
    }
    
    msg_res.set_status(0);
    if(0 == notify_dbs(msg_res, controller) ) {
        MI_AISERVER_LOG(MI_TRACE) << "OUT Inference operation.";
        return 0;
    } else {
        return -1;
    }
}

MED_IMG_END_NAMESPACE