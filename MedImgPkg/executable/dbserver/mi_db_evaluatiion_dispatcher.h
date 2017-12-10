#ifndef MEDIMG_MI_DB_EVALUATION_DISPATCHER_H
#define MEDIMG_MI_DB_EVALUATION_DISPATCHER_H

#include <map>
#include <set>
#include <deque>
#include <string>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include "util/mi_model_interface.h"

#include "mi_db_server_common.h"
#include "util/mi_ipc_common.h"
#include "io/mi_db.h"

MED_IMG_BEGIN_NAMESPACE

class MsgEvaluationResponse;
class DBServerController;

class DBEvaluationDispatcher : public IModel {
public:
    DBEvaluationDispatcher();
    virtual ~DBEvaluationDispatcher();

    void set_controller(std::shared_ptr<DBServerController> controller);

    int request_evaluation(const unsigned int client_id, const std::string& series_id);
    int receive_evaluation(MsgEvaluationResponse* msg_res);

private:
    void add_request(const unsigned int client_id, DB::ImgItem& item);
    
    void notify_all(IPCPackage* pkg);

    bool update_request_series(const std::string& series);
    bool update_receive_series(const std::string& series);

private:
    typedef std::deque<unsigned int> RequesterQueue;
    typedef std::set<unsigned int> RequesterSet;
    struct RequesterCollection {
        RequesterQueue req_queue;
        RequesterSet req_set;
    };
    std::map<std::string, RequesterCollection> _request_queue;

    boost::mutex _mutex_request;
    boost::condition _condition_request;

    boost::mutex _mutex_reveive;
    boost::condition _condition_receive;

    boost::mutex _mutex_series;
    boost::mutex _mutex_queue;

    std::string _request_series;
    std::string _receive_series;

    std::weak_ptr<DBServerController> _controller;

private:
    DISALLOW_COPY_AND_ASSIGN(DBEvaluationDispatcher);
};

MED_IMG_END_NAMESPACE

#endif