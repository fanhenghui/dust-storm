#include "mi_ai_operation_db_request_evaluation.h"

#include <time.h>

#include "util/mi_file_util.h"
#include "util/mi_memory_shield.h"

#include "io/mi_nodule_set.h"
#include "io/mi_nodule_set_parser.h"
#include "io/mi_db.h"
#include "io/mi_protobuf.h"
#include "io/mi_configure.h"

#include "appcommon/mi_app_common_define.h"

#include "mi_ai_server_controller.h"
#include "mi_ai_server_thread_model.h"
#include "mi_ai_lung_evaluate_py_wrapper.h"

MED_IMG_BEGIN_NAMESPACE

AIOpDBRequestEvaluation::AIOpDBRequestEvaluation() {

}

AIOpDBRequestEvaluation::~AIOpDBRequestEvaluation() {

}

int notify_dbs(MsgEvaluationResponse& msg, std::shared_ptr<AIServerController> controller) {
    int msg_size = 0;                
    char* msg_buffer = nullptr;
    if (0 != protobuf_serialize(msg, msg_buffer, msg_size)) {
        MI_AISERVER_LOG(MI_ERROR) << "notify DBS failed.";
        return -1;
    } else {
        IPCDataHeader header;
        header.msg_id = COMMAND_ID_DB_AI_OPERATION;
        header.op_id = OPERATION_ID_DB_AI_EVALUATION_RESULT;
        header.data_len = msg_size;
        msg.Clear();
        controller->get_thread_model()->async_send_data(new IPCPackage(header,msg_buffer));
        return 0;
    }
}

int AIOpDBRequestEvaluation::execute() {
    MI_AISERVER_LOG(MI_TRACE) << "IN AIOpDBRequestEvaluation.";

    //--------------------------------------------------------------//
    //TODO 1 AIS需要做负载均衡，增加和显卡数目相同的可配置的 AI计算队列
    //TODO 2 这里AI算法的接口需要变更 : path换成instance file path 的集合 
    //TODO 3 这里只考虑了一种AI算法，后续得从工厂类得到具体的实例来计算
    //TODO 4 目前是AIS和DBS在一台机器上的部署方法，所以AIS可以直接写文件到磁盘（否则就得穿二进制流回去）
    //--------------------------------------------------------------//

    AISERVER_CHECK_NULL_EXCEPTION(_buffer);

    std::shared_ptr<AIServerController> controller  = get_controller<AIServerController>();
    AISERVER_CHECK_NULL_EXCEPTION(controller);

    //response message
    MsgEvaluationResponse msg_res;

    //----------------------------------//
    // parse request message
    //----------------------------------//
    MsgEvaluationRequest msg_req;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg_req)) {
        MI_AISERVER_LOG(MI_ERROR) << "parse evaluation request message failed.";
        msg_res.set_status(-1);
        msg_res.set_err_msg("parse request message failed.");
        return notify_dbs(msg_res, controller);
    }

    const std::string& series_uid = msg_req.series_uid();
    const int64_t series_pk = msg_req.series_pk();

    const int64_t eva_pk = msg_req.eva_pk();
    const int eva_type = msg_req.eva_type();
    const std::string eva_file_path = msg_req.eva_file_path();

    const int64_t prep_pk = msg_req.prep_pk();
    const int prep_type = msg_req.prep_type();
    const std::string prep_file_path = msg_req.prep_file_path();
    const bool prep_expired = msg_req.prep_expired();

    std::vector<std::string> instance_files(msg_req.instance_files_size());
    for (int i=0; i<msg_req.instance_files_size(); ++i) {
        instance_files[i] = msg_req.instance_files(i);
    }

    msg_req.Clear();

    msg_res.set_series_uid(series_uid);
    msg_res.set_series_pk(series_pk);
    
    msg_res.set_eva_pk(eva_pk);
    msg_res.set_eva_type(eva_type);
    msg_res.set_eva_file_path(eva_file_path);

    msg_res.set_prep_pk(prep_pk);
    msg_res.set_prep_type(prep_type);
    msg_res.set_prep_file_path(prep_file_path);
    msg_res.set_prep_expired(prep_expired);

    //----------------------------------//
    // initialize python run time
    //----------------------------------//
    //解析dicom路径, 为了上面的 TOOD 2 临时写的代码
    std::string dcm_path;
    for (size_t i=eva_file_path.size()-1; i>=0; ++i) {
        if (eva_file_path[i] == '/' || eva_file_path[i] == '\\') {
            dcm_path = eva_file_path.substr(0,i);
        }
    }
    
    //get python running path
    const std::string pytorch_path = Configure::instance()->get_pytorch_path();
    const std::string py_interface_path = Configure::instance()->get_py_interface_path();

    //debug print info
    // MI_AISERVER_LOG(MI_DEBUG) << "IN AI evaluation operation:";
    // MI_AISERVER_LOG(MI_DEBUG) << "pytorch path: " << pytorch_path;
    // MI_AISERVER_LOG(MI_DEBUG) << "py interface path: " << py_interface_path;
    // MI_AISERVER_LOG(MI_DEBUG) << "series ID: " << series_uid;
    // MI_AISERVER_LOG(MI_DEBUG) << "DICOM direction: " << dcm_path;

    //为了TODO1　写的临时代码，反复创建AILungEvaulatePyWrapper实例会发生内存泄露（貌似是python c++接口的问题）
    static std::shared_ptr<medical_ai::AILungEvaulatePyWrapper> wrapper(new medical_ai::AILungEvaulatePyWrapper());
    static bool init_wrapper = false;
    if (!init_wrapper) {
        if(-1 == wrapper->init(pytorch_path.c_str() ,py_interface_path.c_str()) ){
            MI_AISERVER_LOG(MI_FATAL) << "init python env failed.";
            msg_res.set_status(-1);
            msg_res.set_err_msg("init python env failed.");
            //return notify_dbs(msg_res, controller);
        }
        init_wrapper = true;
    }
    // if(-1 == wrapper->init(pytorch_path.c_str() ,py_interface_path.c_str()) ){
    //     MI_AISERVER_LOG(MI_FATAL) << "init python env failed.";
    //     msg_res.set_status(-1);
    //     msg_res.set_err_msg("init python env failed.");
    //     return notify_dbs(msg_res, controller);
    // }
    MI_AISERVER_LOG(MI_DEBUG) << "after initialize.";

    //----------------------------------//
    //　get API version
    //----------------------------------//
    Version ver(1,0,0);
    std::string ver_str(wrapper->get_version());
    if (0 != make_version(ver_str, ver)) {
        MI_AISERVER_LOG(MI_ERROR) << "AI API get invalid version:" << ver_str;
            msg_res.set_status(-1);
            msg_res.set_err_msg("AI API get invalid version.");
            return notify_dbs(msg_res, controller);
    }

    //TODO 这里目前是一个接口，未来最好是两个分开控制的接口
    msg_res.set_prep_version(ver_str);
    msg_res.set_eva_version(ver_str);

    //----------------------------------//
    // preprocess
    //----------------------------------//
    if (prep_expired) {
        MI_AISERVER_LOG(MI_DEBUG) << "AI lung preprocess start.";
        clock_t _start = clock();
        char* buffer_prep_data = nullptr;
        int buffer_prep_data_size = 0;
        if (-1 == wrapper->preprocess(dcm_path.c_str(), buffer_prep_data, buffer_prep_data_size) ){
            const char* err = wrapper->get_last_err();
            MI_AISERVER_LOG(MI_ERROR) << "AI lung preprocess failed :" << err;
            msg_res.set_status(-1);
            msg_res.set_err_msg("AI lung preprocess failed.");
            return notify_dbs(msg_res, controller);
        }
        clock_t _end = clock();
        MI_AISERVER_LOG(MI_DEBUG) << "AI lung preprocess end. cost " << double(_end-_start)/CLOCKS_PER_SEC << " s.";

        MemShield shield(buffer_prep_data);
        if (0 != FileUtil::write_raw(prep_file_path, buffer_prep_data, buffer_prep_data_size)) {
            MI_AISERVER_LOG(MI_ERROR) << "write AI preprocess data to: " << prep_file_path << " failed.";
            msg_res.set_status(-1);
            msg_res.set_err_msg("write AI preprocess data to disk failed.");
            return notify_dbs(msg_res, controller);
        }
    }

    //----------------------------------//
    // evaluate
    //----------------------------------//
    MI_AISERVER_LOG(MI_DEBUG) << "AI lung evalulate start.";
    clock_t _start = clock();
    medical_ai::AILungEvaulatePyWrapper::VPredictedNodules nodules;
    if(-1 == wrapper->evaluate(prep_file_path.c_str(), nodules)) {
        const char* err = wrapper->get_last_err();
        MI_AISERVER_LOG(MI_ERROR) << "evalulate series: " << series_uid << " failed: " << err;
        msg_res.set_status(-1);
        msg_res.set_err_msg("evalulate series failed.");
        return notify_dbs(msg_res, controller);
    }
    clock_t _end = clock();
    MI_AISERVER_LOG(MI_DEBUG) << "AI lung evalulate end. cost " << double(_end-_start)/CLOCKS_PER_SEC << " s.";

    //encode and write to disk
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    for (size_t i=0; i<nodules.size(); ++i) {
        VOISphere voi(Point3(nodules[i].fX,nodules[i].fY,nodules[i].fZ),nodules[i].fRadius*2.0);
        voi.probability = nodules[i].fProb;
        nodule_set->add_nodule(voi);
    }

    NoduleSetParser ns_parser;
    ns_parser.set_series_id(series_uid);
    if(IO_SUCCESS != ns_parser.save_as_csv(eva_file_path, nodule_set)) {
        MI_AISERVER_LOG(MI_ERROR) << "save evaluated result to " << eva_file_path << " failed.";
        msg_res.set_status(-1);
        msg_res.set_err_msg("save evaluated result failed.");
        return notify_dbs(msg_res, controller);
    }
    
    //----------------------------------//
    // done
    //----------------------------------//
    msg_res.set_status(0);
    if(0 == notify_dbs(msg_res, controller) ) {
        MI_AISERVER_LOG(MI_TRACE) << "OUT AIOpDBRequestEvaluation.";
        return 0;
    } else {
        return -1;
    }
}

MED_IMG_END_NAMESPACE