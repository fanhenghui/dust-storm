#include "mi_db_evaluatiion_dispatcher.h"

#include "util/mi_ipc_server_proxy.h"

#include "io/mi_nodule_set_parser.h"
#include "io/mi_nodule_set.h"

#include "appcommon/mi_message.pb.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_operation_interface.h"

#include "mi_db_server_thread_model.h"
#include "mi_db_server_logger.h"
#include "mi_db_server_controller.h"

MED_IMG_BEGIN_NAMESPACE

namespace {
IPCPackage* create_error_message(const std::string& err) {
    MsgString msgErr;
    msgErr.set_context(err);
    const int buffer_size = msgErr.ByteSize();
    if (buffer_size > 0) {
        IPCDataHeader header;
        header.msg_id = COMMAND_ID_BE_DB_SEND_ERROR;
        header.data_len = buffer_size;
        char* buffer = new char[buffer_size];
        if (nullptr != buffer) {
            if (!msgErr.SerializeToArray(buffer, buffer_size)) {
                delete [] buffer;
                buffer = nullptr;
            } else {
                return new IPCPackage(header, buffer);;
            }
        }
    }
    return nullptr;
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

    virtual std::shared_ptr<IOperation> create() {
        return std::shared_ptr<DBOpRequestEvaluation>(new DBOpRequestEvaluation());
    }
};

}

DBEvaluationDispatcher::DBEvaluationDispatcher() {

}

DBEvaluationDispatcher::~DBEvaluationDispatcher() {

}

void DBEvaluationDispatcher::set_controller(std::shared_ptr<DBServerController> controller) {
    _controller = controller;
}

int DBEvaluationDispatcher::request_evaluation(const unsigned int client_id, const std::string& series_id) {
    // lock
    ConditionGuard guard(_condition_request);
    struct SeriesCleaner {
        DBEvaluationDispatcher* _dispatcher;
        SeriesCleaner(DBEvaluationDispatcher* dispatcher):_dispatcher(dispatcher){}
        ~SeriesCleaner() { _dispatcher->update_request_series("");}
    } cleaner(this);

    if (update_request_series(series_id)) {
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
    DB::ImgItem item;
    if(0 != db->get_dcm_item(series_id, item) ) {
        SEND_ERROR_TO_BE(server_proxy, client_id, "DICOM series item not existed.");
        return -1;
    }

    if (item.annotation_ai_path.empty()) {
        add_request(client_id, item);     
        return 0;
    }

    //TODO check version

    NoduleSetParser parser;
    parser.set_series_id(series_id);
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    if( IO_SUCCESS != parser.load_as_csv(item.annotation_ai_path, nodule_set) ) {
        SEND_ERROR_TO_BE(server_proxy, client_id, "load annotation file failed.");
        return -1;
    }

    MsgAnnotationCollectionDB msgAnnos;
    msgAnnos.set_series_uid(series_id);

    const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
    for (auto it = vois.begin(); it != vois.end(); ++it) {
        const VOISphere &voi = *it;
        MsgAnnotationUnitDB* anno = msgAnnos.add_annotation();
        anno->set_x(voi.center.x);
        anno->set_y(voi.center.y);        
        anno->set_z(voi.center.z);        
        anno->set_r(voi.diameter);
        anno->set_p(voi.para0);
    }

    const int buffer_size = msgAnnos.ByteSize();
    char* buffer = new char[buffer_size];
    if (!msgAnnos.SerializeToArray(buffer, buffer_size)) {
        SEND_ERROR_TO_BE(server_proxy, client_id, "serialize message for AI annotation failed.");
        delete [] buffer;
        return -1;
    }
    msgAnnos.Clear();
    
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
    const std::string series_id = msg_res->series_uid();
    const std::string ai_anno_path = msg_res->ai_anno_path();
    const std::string ai_im_data_path = msg_res->ai_im_data_path();

    if (update_receive_series(series_id)){
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

    //update DB
    bool recal_im_data = msg_res->recal_im_data();
    if (recal_im_data) {
        if (!ai_im_data_path.empty()){
            db->update_ai_intermediate_data(series_id, ai_im_data_path);
        } else {
            MI_DBSERVER_LOG(MI_ERROR) << "update empty AI intermediate data path.";
        }
    }
    
    if (ai_anno_path.empty()){
        IPCPackage* err_pkg = create_error_message("update empty AI annotation data path.");
        notify_all(err_pkg);
        return -1;
    }
    
    db->update_ai_annotation(series_id, ai_anno_path);

    //load annotation and send to client
    NoduleSetParser parser;
    parser.set_series_id(series_id);
    std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
    if( IO_SUCCESS != parser.load_as_csv(ai_anno_path, nodule_set) ) {
        IPCPackage* err_pkg = create_error_message("load evaluation result file failed.");
        notify_all(err_pkg);
        return -1;
    }

    MsgAnnotationCollectionDB msgAnnos;
    msgAnnos.set_series_uid(series_id);
    const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
    for (auto it = vois.begin(); it != vois.end(); ++it) {
        const VOISphere &voi = *it;
        MsgAnnotationUnitDB* anno = msgAnnos.add_annotation();
        anno->set_x(voi.center.x);
        anno->set_y(voi.center.y);        
        anno->set_z(voi.center.z);        
        anno->set_r(voi.diameter);
        anno->set_p(voi.para0);
    }
    const int buffer_size = msgAnnos.ByteSize();
    char* buffer = new char[buffer_size];
    if (!msgAnnos.SerializeToArray(buffer, buffer_size)) {
        IPCPackage* err_pkg = create_error_message("serialize message for AI annotation failed.");
        notify_all(err_pkg);
        delete [] buffer;
        return -1;
    }
    msgAnnos.Clear();
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_BE_DB_SEND_AI_EVALUATION;
    header.data_len = buffer_size;
    IPCPackage* res_pkg = new IPCPackage(header,buffer); 
    notify_all(res_pkg);

    return 0;
}

void DBEvaluationDispatcher::add_request(const unsigned int client_id, DB::ImgItem& item) {
    boost::mutex::scoped_lock locker_queue(_mutex_queue);
    const std::string series_id = item.series_id;
    auto it = _request_queue.find(series_id);
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
        RequesterCollection req_coll;
        req_coll.req_set.insert(client_id);
        req_coll.req_queue.push_back(client_id);
        _request_queue.insert(std::make_pair(series_id, req_coll));

        MI_DBSERVER_LOG(MI_INFO) << "Ecaluation dispatcher: trigger evaluation " << client_id;

        //send evaluation request to AI server
        std::shared_ptr<DBServerController> controller = _controller.lock();
        DBSERVER_CHECK_NULL_EXCEPTION(controller);

        MsgEvaluationRequest msg;
        msg.set_series_uid(series_id);
        msg.set_dcm_path(item.dcm_path);
        msg.set_ai_anno_path(item.dcm_path+"/"+series_id+".csv");
        msg.set_client_socket_id(client_id);
        if (item.ai_intermediate_data_path.empty()) {
            msg.set_ai_im_data_path(item.dcm_path+"/"+series_id+".npy");
            msg.set_recal_im_data(true);
        } else {
            msg.set_ai_im_data_path(item.ai_intermediate_data_path);
            msg.set_recal_im_data(false);
        }
        int msg_buffer_size = msg.ByteSize();
        char* msg_buffer = new char[msg_buffer_size];
        if (msg_buffer_size != 0 && msg.SerializeToArray(msg_buffer,msg_buffer_size)){
            IPCDataHeader header;
            header.data_len = msg_buffer_size;
            header.receiver = controller->get_ais_client();
            header.msg_id = COMMAND_ID_AI_DB_OPERATION;
            header.op_id = OPERATION_ID_AI_DB_REQUEST_AI_EVALUATION;
            std::shared_ptr<DBOpRequestEvaluation> op(new DBOpRequestEvaluation());
            op->set_data(header, msg_buffer);
            op->set_controller(controller);
            controller->get_thread_model()->push_operation_ais(op);
        } 
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