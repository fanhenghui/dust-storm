#include "mi_db_evaluatiion_dispatcher.h"

#include "util/mi_ipc_server_proxy.h"
#include "util/mi_operation_interface.h"

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
    // lock
    ConditionGuard guard(_condition_request);
    struct SeriesCleaner {
        DBEvaluationDispatcher* _dispatcher;
        SeriesCleaner(DBEvaluationDispatcher* dispatcher):_dispatcher(dispatcher){}
        ~SeriesCleaner() { _dispatcher->update_request_series("");}
    } cleaner(this);

    const std::string series_uid = msg_req->series_uid();
    const int64_t series_pk = msg_req->series_pk();

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

    //query DB
    PreprocessInfo prep_key;
    prep_key.prep_type = LUNG_AI_INTERMEDIATE_DATA;
    prep_key.series_fk = series_pk;
    std::vector<PreprocessInfo> ai_prep_infos;
    if (0 != db->query_preprocess(prep_key, &ai_prep_infos)) {
        SEND_ERROR_TO_BE(server_proxy, client_id, "query db ai preprocess failed.");
        return -1;
    }
    int64_t ai_prep_pk = ai_prep_infos.empty() ? -1 : ai_prep_infos[0].id;

    EvaluationInfo eva_key;
    eva_key.series_fk = series_pk;
    std::vector<EvaluationInfo> eva_infos;
    if (0 != db->query_evaluation(eva_key, &eva_infos)) {
        SEND_ERROR_TO_BE(server_proxy, client_id, "query db evaluation failed.");
        return -1;
    }
    int64_t eva_pk = eva_infos.empty() ? -1 : eva_infos[0].id;
    
    //TODO check version
    if (eva_infos.empty()) {
        add_request(client_id, msg_req, eva_pk, ai_prep_pk);     
        return 0;
    }
    
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
    
    return 0;
}

int DBEvaluationDispatcher::receive_evaluation(MsgEvaluationResponse* msg_res) {
    //lock
    ConditionGuard guard(_condition_receive);
    struct SeriesCleaner {
        DBEvaluationDispatcher* _dispatcher;
        SeriesCleaner(DBEvaluationDispatcher* dispatcher):_dispatcher(dispatcher){}
        ~SeriesCleaner() { _dispatcher->update_receive_series("");}
    } cleaner(this);

    const int status = msg_res->status();
    const std::string series_uid = msg_res->series_uid();
    const int64_t series_fk = msg_res->series_fk();
    const int64_t eva_pk = msg_res->eva_pk();
    const int64_t prep_pk = msg_res->prep_pk();
    const std::string ai_eva_file_path = msg_res->ai_eva_file_path();
    const std::string ai_im_data_path = msg_res->ai_im_data_path();

    if (update_receive_series(series_uid)){
        boost::mutex::scoped_lock _mutex_request;
        _condition_request.wait(_mutex_request);
    }

    DBSERVER_CHECK_NULL_EXCEPTION(msg_res);

    std::shared_ptr<DBServerController> controller = _controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<DB> db = controller->get_db();
    DBSERVER_CHECK_NULL_EXCEPTION(db);

    //check error
    if (status == -1) {
        std::string err_msg = msg_res->err_msg();
        IPCPackage* err_pkg = create_error_message(err_msg);
        notify_all(err_pkg);
        return -1;
    }


    bool recal_im_data = msg_res->recal_im_data();
    if (recal_im_data) {
        PreprocessInfo info;
        info.series_fk = series_fk;
        info.prep_type = LUNG_AI_INTERMEDIATE_DATA;
        info.file_path = ai_im_data_path;
        info.version = "0.0.0";
        int64_t file_size = -1;
        if(0 != FileUtil::get_file_size(ai_im_data_path, file_size) ) {
            IPCPackage* err_pkg = create_error_message("get ai preprocess data file size failed.");
            notify_all(err_pkg);
            return -1;
        }
        info.file_size = file_size;

        if (prep_pk < 1) {
            //insert new one
            if (0 != db->insert_preprocess(info)) {
                IPCPackage* err_pkg = create_error_message("insert ai preprocess data failed.");
                notify_all(err_pkg);
                return -1;
            }
        } else {
            info.id = prep_pk;
            //update old one 
            if (0 != db->update_preprocess(info)) {
                IPCPackage* err_pkg = create_error_message("update ai preprocess data failed.");
                notify_all(err_pkg);
                return -1;
            }
        }
    }
    
    if (ai_eva_file_path.empty()){
        IPCPackage* err_pkg = create_error_message("update empty AI annotation data path.");
        notify_all(err_pkg);
        return -1;
    }
    
    EvaluationInfo info;
    info.series_fk = series_fk;
    info.eva_type = LUNG_NODULE;
    info.version = "0.0.0";
    info.file_path = ai_eva_file_path;
    int64_t file_size = -1;
    if(0 != FileUtil::get_file_size(ai_eva_file_path, file_size) ) {
        IPCPackage* err_pkg = create_error_message("get ai evaluation file size failed.");
        notify_all(err_pkg);
        return -1;
    }
    info.file_size = file_size;

    if (eva_pk < 1) {
        //insert new one
        if (0 != db->insert_evaluation(info)) {
            IPCPackage* err_pkg = create_error_message("insert ai evaluation failed.");
            notify_all(err_pkg);
            return -1;
        }
    } else {
        info.id = prep_pk;
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
    if( IO_SUCCESS != parser.load_as_csv(ai_eva_file_path, nodule_set) ) {
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

void DBEvaluationDispatcher::add_request(const unsigned int client_id, MsgEvaluationRetrieveKey* msg_req, int64_t eva_pk, int64_t ai_prep_pk) {
    boost::mutex::scoped_lock locker_queue(_mutex_queue);

    std::shared_ptr<DBServerController> controller = _controller.lock();
    DBSERVER_CHECK_NULL_EXCEPTION(controller);
    std::shared_ptr<DB> db = controller->get_db();
    DBSERVER_CHECK_NULL_EXCEPTION(db);

    const int64_t series_pk = msg_req->series_pk();
    const std::string& series_uid = msg_req->series_uid();
    const std::string& study_uid = msg_req->study_uid();
    const std::string db_path = Configure::instance()->get_db_path();
    const std::string series_path = db_path + "/" + study_uid + "/" + series_uid + "/";
    const std::string file_path = series_path + series_uid + ".csv";
    const std::string ai_im_file_path = series_path + series_uid + ".npy";

    
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
        
        //query AI intermidiate data
        PreprocessInfo prep_key;
        prep_key.series_fk = series_pk;
        prep_key.prep_type = LUNG_AI_INTERMEDIATE_DATA;
        std::vector<PreprocessInfo> prep_infos;
        if (0 != db->query_preprocess(prep_key, &prep_infos)) {
            MI_DBSERVER_LOG(MI_ERROR) << "query evaluation failed.";
            return;
        }

        MsgEvaluationRequest msg;
        msg.set_series_uid(series_uid);
        msg.set_eva_pk(eva_pk);
        msg.set_prep_pk(prep_pk);
        msg.set_ai_eva_file_path(file_path);
        msg.set_ai_im_data_path(ai_im_file_path);
        msg.set_client_socket_id(client_id);
        if (prep_infos.empty()) {
            msg.set_recal_im_data(true);

            std::vector<std::string> instances_file_paths;
            if (0 != db->query_series_instance(series_pk, &instances_file_paths)) {
                MI_DBSERVER_LOG(MI_ERROR) << "query series instance failed.";
                return;
            }
            for (size_t i = 0; i< instances_file_paths; ++i) {
               std::string* files = msg.add_instance_files(); 
               *files = instances_file_paths[i];
            }
        } else {
            
            msg.set_recal_im_data(false);
        }
        int msg_buffer_size = 0;
        char* msg_buffer = nullptr;
        if (0 != protobuf_serialize(msg, msg_buffer, msg_buffer_size)) {
        }

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