#include "mi_db_evaluatiion_dispatcher.h"

#include "util/mi_ipc_server_proxy.h"
#include "util/mi_operation_interface.h"
#include "util/mi_file_util.h"
#include "util/mi_version_util.h"

#include "io/mi_nodule_set_parser.h"
#include "io/mi_nodule_set.h"
#include "io/mi_protobuf.h"
#include "io/mi_configure.h"

#include "appcommon/mi_app_common_define.h"

#include "mi_db_server_thread_model.h"
#include "mi_db_server_logger.h"
#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
IPCPackage* create_error_message(const std::string& err) {
    MsgString msg_err;
    msg_err.set_context(err);
    char* buffer = nullptr;
    int buffer_size = 0;
    if (0 != protobuf_serialize(msg_err, buffer, buffer_size)) {
        msg_err.Clear();
        IPCDataHeader header;
        header.msg_id = COMMAND_ID_BE_DB_SEND_ERROR;
        header.data_len = buffer_size;
        return new IPCPackage(header, buffer);
    } else {
        return nullptr;
    }
}

struct ConditionGuard {
    boost::condition* _condition;

    ConditionGuard(boost::condition& condition):_condition(&condition) {
    }

    ~ConditionGuard() {
        _condition->notify_one();
    }
};

class DBOpRequestEvaluation: public IOperation {
public:
    DBOpRequestEvaluation() {}
    virtual ~DBOpRequestEvaluation() {}

    virtual int execute() {
        DBSERVER_CHECK_NULL_EXCEPTION(_buffer);
    
        std::shared_ptr<DBServerController> controller  = get_controller<DBServerController>();
        DBSERVER_CHECK_NULL_EXCEPTION(controller);
        std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_ais();

        IPCPackage* package = new IPCPackage(_header, _buffer);
        _buffer = nullptr;//move op buffer to IPC package

        if (0 != server_proxy->async_send_data(package)) {
            delete package;
            package = nullptr;
            MI_DBSERVER_LOG(MI_WARNING) << "send request AI evaluation faield.(client disconnected)";
        }

        return 0;
    }

    CREATE_EXTENDS_OP(DBOpRequestEvaluation)
};

}

DBEvaluationDispatcher::DBEvaluationDispatcher() {

}

DBEvaluationDispatcher::~DBEvaluationDispatcher() {

}

void DBEvaluationDispatcher::set_controller(std::shared_ptr<DBServerController> controller) {
    _controller = controller;
}

int DBEvaluationDispatcher::request_evaluation(const unsigned int client_id, MsgEvaluationRetrieveKey* msg_req) {

    //-------------------------------------------------------//
    // request evaluation procedure:
    // query evaluation 
    //     null -> send request to AIS to calculate: 1 AI preprocess data; 2 eva
    //     has a result -> check evalution version
    //                         <  current version: check AI preprocess version , and send compute request to AIS
    //                         == current version: send result back to BE directly
    //-------------------------------------------------------//
    
    // lock
    ConditionGuard guard(_condition_request);
    struct SeriesCleaner {
        DBEvaluationDispatcher* _dispatcher;
        SeriesCleaner(DBEvaluationDispatcher* dispatcher):_dispatcher(dispatcher){}
        ~SeriesCleaner() { _dispatcher->update_request_series("");}
    } cleaner(this);

    const std::string& series_uid = msg_req->series_uid();
    const std::string& study_uid = msg_req->study_uid();
    const int64_t series_pk = msg_req->series_pk();
    const int eva_type = msg_req->eva_type();
    const std::string db_path = Configure::instance()->get_db_path();
    const std::string series_data_path = db_path + "/" + study_uid + "/" + series_uid + "/";
    const std::string eva_file_path = series_data_path + series_uid + ".csv";
    const std::string prep_file_path = series_data_path + series_uid + ".npy";
    
    //-----------------------------------------------------------------------//
    //TODO 这里会汇聚各种形式的中间数据，根据evaluation的类型，选择对应AI中间(预处理)类型
    //     不过目前只支持一种evaluation对应一个AI中间(预处理)数据
    //-----------------------------------------------------------------------//
    int prep_type = -1;
    if (eva_type == LUNG_NODULE) {
        prep_type = LUNG_AI_DATA;
    }

    if (update_request_series(series_uid)) {
        boost::mutex::scoped_lock request_locker(_mutex_reveive);
        _condition_receive.wait(_mutex_reveive);
    }
    
    std::shared_ptr<DBServerController> controller = _controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<DB> db = controller->get_db();
    DBSERVER_CHECK_NULL_EXCEPTION(db);

    std::shared_ptr<DBServerThreadModel> thread_model = controller->get_thread_model();
    DBSERVER_CHECK_NULL_EXCEPTION(thread_model);

    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);

    //---------------------------------//
    // query evaluation & ai preprocess
    //---------------------------------//
    PreprocessInfo prep_key;
    prep_key.prep_type = prep_type;
    prep_key.series_fk = series_pk;
    std::vector<PreprocessInfo> prep_infos;
    if (0 != db->query_preprocess(prep_key, &prep_infos)) {
        SEND_ERROR_TO_BE(server_proxy, client_id, "query db ai preprocess failed.");
        return -1;
    }

    EvaluationInfo eva_key;
    eva_key.series_fk = series_pk;
    std::vector<EvaluationInfo> eva_infos;
    if (0 != db->query_evaluation(eva_key, &eva_infos)) {
        SEND_ERROR_TO_BE(server_proxy, client_id, "query db evaluation failed.");
        return -1;
    }

    int64_t eva_pk = eva_infos.empty() ? -1 : eva_infos[0].id;
    int64_t prep_pk = prep_infos.empty() ? -1 : prep_infos[0].id;

    //---------------------------------//
    // check version
    //---------------------------------//
    bool eva_expired = false;
    bool prep_expired = false;
    EvaluationInfo eva_info;
    PreprocessInfo prep_info;
    if (eva_pk > 0) {
        //check evaluation version
        Version eva_version;
        if (-1 == make_version(eva_infos[0].version, eva_version)) {
            SEND_ERROR_TO_BE(server_proxy, client_id, "query db ai evaluation failed: invalid version record.");
            return -1;
        }
        eva_expired = eva_version < Configure::instance()->get_evaluation_version((EvaluationType)eva_type);
        
        //只有在evaluation 失效的情况下才考虑 preprocess
        if (eva_expired) {
            eva_info = eva_infos[0];
            if (prep_pk < 1) {
                //这种情况理论上不会发生，写这个分支是防止数据意外: 即evaluation 有且失效 但是preprocess不存在（被删除）
                prep_info.series_fk = series_pk;
                prep_info.prep_type = prep_type;
                prep_info.file_path = prep_file_path;
            } else {
                prep_info = prep_infos[0];
                //check preprocess version
                Version prep_version;
                if (-1 == make_version(prep_infos[0].version, prep_version)) {
                    SEND_ERROR_TO_BE(server_proxy, client_id, "query db ai evaluation failed: invalid version record.");
                    return -1;
                }
                prep_expired = prep_version < Configure::instance()->get_preprocess_version((PreprocessType)prep_type);
            }
        }
    } else {
        eva_expired = true;
        prep_expired = true;

        eva_info.series_fk = series_pk;
        eva_info.eva_type = eva_type;
        eva_info.file_path = eva_file_path;

        prep_info.series_fk = series_pk;
        prep_info.prep_type = prep_type;
        prep_info.file_path = prep_file_path;
        
    }
    
    if (eva_expired) {
        add_request(client_id, msg_req, eva_info, prep_info, prep_expired);     
    } else {
        NoduleSetParser parser;
        parser.set_series_id(series_uid);
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        if( IO_SUCCESS != parser.load_as_csv(eva_infos[0].file_path, nodule_set) ) {
            SEND_ERROR_TO_BE(server_proxy, client_id, "load evaluation file failed.");
            return -1;
        }

        MsgAnnotationCollectionDB msg_annos;
        msg_annos.set_series_uid(series_uid);

        const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
        for (auto it = vois.begin(); it != vois.end(); ++it) {
            const VOISphere &voi = *it;
            MsgAnnotationUnitDB* anno = msg_annos.add_annotation();
            anno->set_x(voi.center.x);
            anno->set_y(voi.center.y);        
            anno->set_z(voi.center.z);        
            anno->set_r(voi.diameter);
            anno->set_p(voi.probability);
        }

        int buffer_size = 0;
        char* buffer = nullptr;
        if (0 != protobuf_serialize(msg_annos, buffer, buffer_size)) {
            SEND_ERROR_TO_BE(server_proxy, client_id, "serialize message for AI annotation failed.");
            return -1;
        }
        msg_annos.Clear();
        
        IPCDataHeader header;
        header.receiver = client_id;
        header.msg_id = COMMAND_ID_BE_DB_SEND_AI_EVALUATION;
        header.data_len = buffer_size;
        IPCPackage* package = new IPCPackage(header,buffer); 
        if(0 != server_proxy->async_send_data(package)) {
            delete package;
            package = nullptr;
            MI_DBSERVER_LOG(MI_WARNING) << "send AI annotation to client failed.(client disconnected)";
            return -1;
        }
    }

    return 0;
}

int DBEvaluationDispatcher::receive_evaluation(MsgEvaluationResponse* msg_res) {
    //-------------------------------------------------------//
    // receive evaluation procedure:
    // check error from AIS
    // insert / update AI preprocess if expired
    // insert / update AI evaluation
    //-------------------------------------------------------//

    //lock
    ConditionGuard guard(_condition_receive);
    struct SeriesCleaner {
        DBEvaluationDispatcher* _dispatcher;
        SeriesCleaner(DBEvaluationDispatcher* dispatcher):_dispatcher(dispatcher){}
        ~SeriesCleaner() { _dispatcher->update_receive_series("");}
    } cleaner(this);

    const int status = msg_res->status();
    const std::string series_uid = msg_res->series_uid();
    const int64_t series_fk = msg_res->series_pk();

    if (update_receive_series(series_uid)){
        boost::mutex::scoped_lock _mutex_request;
        _condition_request.wait(_mutex_request);
    }

    DBSERVER_CHECK_NULL_EXCEPTION(msg_res);

    std::shared_ptr<DBServerController> controller = _controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<DB> db = controller->get_db();
    DBSERVER_CHECK_NULL_EXCEPTION(db);

    //---------------------------------//
    //check error
    //---------------------------------//
    if (status == -1) {
        std::string err_msg = msg_res->err_msg();
        IPCPackage* err_pkg = create_error_message(err_msg);
        notify_all(err_pkg);
        return -1;
    }


    bool prep_expired = msg_res->prep_expired();
    if (prep_expired) {
        PreprocessInfo info;
        info.id = msg_res->prep_pk();
        info.series_fk = series_fk;
        info.prep_type = msg_res->prep_type();
        info.file_path = msg_res->prep_file_path();
        info.version = msg_res->prep_version();
        int64_t file_size = -1;
        if(0 != FileUtil::get_file_size(info.file_path, file_size) ) {
            IPCPackage* err_pkg = create_error_message("get ai preprocess data file size failed.");
            notify_all(err_pkg);
            return -1;
        }
        info.file_size = file_size;

        if (msg_res->prep_pk() < 1) {
            //insert new one
            if (0 != db->insert_preprocess(info)) {
                IPCPackage* err_pkg = create_error_message("insert ai preprocess data failed.");
                notify_all(err_pkg);
                return -1;
            }
        } else {
            //update old one 
            if (0 != db->update_preprocess(info)) {
                IPCPackage* err_pkg = create_error_message("update ai preprocess data failed.");
                notify_all(err_pkg);
                return -1;
            }
        }
    }
    
    EvaluationInfo info;
    info.id = msg_res->eva_pk();
    info.series_fk = series_fk;
    info.eva_type = msg_res->eva_type();
    info.file_path = msg_res->eva_file_path();
    info.version = msg_res->eva_version();
    int64_t file_size = -1;
    if(0 != FileUtil::get_file_size(info.file_path, file_size) ) {
        IPCPackage* err_pkg = create_error_message("get ai evaluation file size failed.");
        notify_all(err_pkg);
        return -1;
    }
    info.file_size = file_size;

    if (msg_res->eva_pk() < 1) {
        //insert new one
        if (0 != db->insert_evaluation(info)) {
            IPCPackage* err_pkg = create_error_message("insert ai evaluation failed.");
            notify_all(err_pkg);
            return -1;
        }
    } else {
        //update old one 
        if (0 != db->update_evaluation(info)) {
            IPCPackage* err_pkg = create_error_message("update ai evaluation failed.");
            notify_all(err_pkg);
            return -1;
        }
    }

    //load annotation and send to client
    NoduleSetParser parser;
    parser.set_series_id(series_uid);
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    if( IO_SUCCESS != parser.load_as_csv(msg_res->eva_file_path(), nodule_set) ) {
        IPCPackage* err_pkg = create_error_message("load evaluation result file failed.");
        notify_all(err_pkg);
        return -1;
    }

    MsgAnnotationCollectionDB msg_annos;
    msg_annos.set_series_uid(series_uid);
    const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
    for (auto it = vois.begin(); it != vois.end(); ++it) {
        const VOISphere &voi = *it;
        MsgAnnotationUnitDB* anno = msg_annos.add_annotation();
        anno->set_x(voi.center.x);
        anno->set_y(voi.center.y);        
        anno->set_z(voi.center.z);        
        anno->set_r(voi.diameter);
        anno->set_p(voi.probability);
    }
    int buffer_size = 0;
    char* buffer = nullptr;
    if (0 != protobuf_serialize(msg_annos, buffer, buffer_size)) {
        IPCPackage* err_pkg = create_error_message("serialize message for AI annotation failed.");
        notify_all(err_pkg);
        return -1;
    }
    msg_annos.Clear();

    IPCDataHeader header;
    header.msg_id = COMMAND_ID_BE_DB_SEND_AI_EVALUATION;
    header.data_len = buffer_size;
    IPCPackage* res_pkg = new IPCPackage(header,buffer); 
    notify_all(res_pkg);

    return 0;
}

void DBEvaluationDispatcher::add_request(const unsigned int client_id, MsgEvaluationRetrieveKey* msg_req, 
    const EvaluationInfo eva_info, const PreprocessInfo& prep_info, bool prep_expired) {
    boost::mutex::scoped_lock locker_queue(_mutex_queue);

    std::shared_ptr<DBServerController> controller = _controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<DB> db = controller->get_db();
    DBSERVER_CHECK_NULL_EXCEPTION(db);

    const int64_t series_pk = msg_req->series_pk();
    const std::string& series_uid = msg_req->series_uid();

    
    auto it = _request_queue.find(series_uid);
    if (it != _request_queue.end()) {
        RequesterCollection& req_coll = it->second;
        if (req_coll.req_set.find(client_id) == req_coll.req_set.end()) {
            req_coll.req_set.insert(client_id);
            req_coll.req_queue.push_back(client_id);
            MI_DBSERVER_LOG(MI_INFO) << "Ecaluation dispatcher: add to evaluation request queue " << client_id;
        } else {
            MI_DBSERVER_LOG(MI_INFO) << "Ecaluation dispatcher: evaluation request already in queue " << client_id;
        }
    } else {
        //first request
        MI_DBSERVER_LOG(MI_INFO) << "Ecaluation dispatcher: trigger evaluation " << client_id;

        //send evaluation request to AI server
        MsgEvaluationRequest msg;

        msg.set_series_uid(series_uid);
        msg.set_series_pk(series_pk);

        msg.set_eva_pk(eva_info.id);
        msg.set_eva_type(eva_info.eva_type);
        msg.set_eva_file_path(eva_info.file_path);
        
        msg.set_prep_pk(prep_info.id);
        msg.set_prep_type(prep_info.prep_type);
        msg.set_prep_file_path(prep_info.file_path);
        msg.set_prep_expired(prep_expired);

        if (prep_expired) {
            std::vector<std::string> instances_file_paths;
            if (0 != db->query_series_instance(series_pk, &instances_file_paths)) {
                MI_DBSERVER_LOG(MI_ERROR) << "query series instance failed.";
                return;
            }
            for (size_t i = 0; i< instances_file_paths.size(); ++i) {
               std::string* files = msg.add_instance_files(); 
               *files = instances_file_paths[i];
            }
        }

        int msg_buffer_size = 0;
        char* msg_buffer = nullptr;
        if (0 == protobuf_serialize(msg, msg_buffer, msg_buffer_size)) {
            //add to queue
            RequesterCollection req_coll;
            req_coll.req_set.insert(client_id);
            req_coll.req_queue.push_back(client_id);
            _request_queue.insert(std::make_pair(series_uid, req_coll));

            //send request to AI server
            IPCDataHeader header;
            header.data_len = msg_buffer_size;
            header.receiver = controller->get_ais_client();
            header.msg_id = COMMAND_ID_AI_DB_OPERATION;
            header.op_id = OPERATION_ID_AI_DB_REQUEST_AI_EVALUATION;
            std::shared_ptr<DBOpRequestEvaluation> op(new DBOpRequestEvaluation());
            op->set_data(header, msg_buffer);
            op->set_controller(controller);
            controller->get_thread_model()->push_operation_ais(op);


            msg.Clear();
            MI_DBSERVER_LOG(MI_INFO) << "send request to AIS.";
        } else {
            MI_DBSERVER_LOG(MI_ERROR) << "add request to DBS evaluation dispatcher failed.";
        }
    }
}

void DBEvaluationDispatcher::notify_all(IPCPackage* pkg) {
    std::shared_ptr<DBServerController> controller = _controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<IPCServerProxy> server_proxy = controller->get_server_proxy_be();
    DBSERVER_CHECK_NULL_EXCEPTION(server_proxy);

    //send message to notify all requester(BE)
    int notify_num = 0;
    {
        boost::mutex::scoped_lock locker(_mutex_queue);

        auto it = _request_queue.find(_receive_series);
        if (it == _request_queue.end()) {
            MI_DBSERVER_LOG(MI_ERROR) << "DB AI dispacher notify null requester.";
            return;
        }
        if (nullptr == pkg) {
            MI_DBSERVER_LOG(MI_ERROR) << "DB AI dispacher notify null package.";
            return;
        }

        RequesterCollection& req_coll = it->second;
        
        for (auto requester = req_coll.req_queue.begin(); requester != req_coll.req_queue.end(); ++requester ) {
            const unsigned int client_id = *requester;
            pkg->header.receiver = client_id;
            if(0 != server_proxy->async_send_data(pkg->clone())) {
                MI_DBSERVER_LOG(MI_WARNING) << "send evaluation to client failed.(client disconnected)";
                continue;
            }
            MI_DBSERVER_LOG(MI_INFO) << "send evaluation to " << client_id;
            ++notify_num;
        }
        //clear queue
        _request_queue.erase(it);
    }
    
    delete pkg;
    pkg = nullptr;
}

bool DBEvaluationDispatcher::update_request_series(const std::string& series) {
    boost::mutex::scoped_lock locker(_mutex_series);
    _request_series = series;
    return _receive_series == _request_series;
}

bool DBEvaluationDispatcher::update_receive_series(const std::string& series) {
    boost::mutex::scoped_lock locker(_mutex_series);
    _receive_series = series;
    return _receive_series == _request_series;
}

MED_IMG_END_NAMESPACE