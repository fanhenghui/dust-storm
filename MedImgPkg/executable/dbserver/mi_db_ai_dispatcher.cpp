#include "mi_db_ai_dispatcher.h"

#include "util/mi_ipc_server_proxy.h"
#include "mi_db_server_thread_model.h"

MED_IMG_BEGIN_NAMESPACE

DBAIDispatcher::DBAIDispatcher() {

}

DBAIDispatcher::~DBAIDispatcher() {

}

void DBAIDispatcher::add_request(const unsigned int requester, const std::string& series_id) {
    boost::mutex::scoped_lock locker(_mutex);
    
    auto it = _request_queue.find(series_id);
    if (it != _request_queue.end()) {
        RequesterCollection& req_coll = it->second;
        if (req_coll.req_set.find(requester) != req_coll.req_set.end()) {
            req_coll.req_set.insert(requester);
            req_coll.req_queue.push_back(requester);
        }
    } else {
        RequesterCollection req_coll;
        req_coll.req_set.insert(requester);
        req_coll.req_queue.push_back(requester);
        _request_queue.insert(std::make_pair(series_id, req_coll));
    }
}

int DBAIDispatcher::notify_unlock(std::shared_ptr<IPCServerProxy> server_proxy, IPCPackage* pkg) {
    boost::mutex::scoped_lock locker(_mutex);

    auto it = _request_queue.find(_locked_series);
    if (it == _request_queue.end()) {
        MI_DBSERVER_LOG(MI_ERROR) << "DB AI dispacher notify null requester.";
        return -1;
    }
    if (nullptr == pkg) {
        MI_DBSERVER_LOG(MI_ERROR) << "DB AI dispacher notify null package.";
        return -1;
    }

    //send message to notify all requester(BE)
    int notify_num = 0;
    RequesterCollection& req_coll = it->second;
    for (auto requester = req_coll.req_queue.begin(); requester != req_coll.req_queue.end(); ++requester ) {
        const unsigned int socket_id = *requester;
        pkg->header.receiver = socket_id;
        if(0 != server_proxy->async_send_data(pkg)) {
            MI_DBSERVER_LOG(MI_WARNING) << "send AI annotation to client failed.(client disconnected)";
            continue;
        }
        ++notify_num;
    }
    if (0 == notify_num) {
        delete pkg;
        pkg = nullptr;
    }
    _locked_series = "";
}

void DBAIDispatcher::lock(const std::string& series_id) {
    boost::mutex::scoped_lock locker(_mutex);
    _locked_series = series_id;
}

void DBAIDispatcher::wait(const std::string& series_id) {
    boost::mutex::scoped_lock locker(_mutex);
    if (series_id == _locked_series) {
        boost::mutex::scoped_lock locker(_mutex_lock_series);
    }
}

MED_IMG_END_NAMESPACE