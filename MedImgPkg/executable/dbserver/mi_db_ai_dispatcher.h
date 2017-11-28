#ifndef MEDIMG_MI_DB_AI_DISPATCHER_H
#define MEDIMG_MI_DB_AI_DISPATCHER_H

#include <map>
#include <set>
#include <deque>
#include <string>
#include <boost/thread/mutex.hpp>
#include "mi_db_server_common.h"
#include "util/mi_model_interface.h"
#include "util/mi_ipc_common.h"

MED_IMG_BEGIN_NAMESPACE

class IPCServerProxy;
class DBAIDispatcher : public IModel {
public:
    DBAIDispatcher();
    virtual ~DBAIDispatcher();

    void add_request(const unsigned int requester_socket_id, const std::string& series_id);
    void lock(const std::string& series_id);
    void wait(const std::string& series_id);
    int notify_unlock(std::shared_ptr<IPCServerProxy> server_proxy, IPCPackage* pkg);

private:
    typedef std::deque<unsigned int> RequesterQueue;
    typedef std::set<unsigned int> RequesterSet;
    struct RequesterCollection {
        RequesterQueue req_queue;
        RequesterSet req_set;
    };
    std::map<std::string, RequesterCollection> _request_queue;
    boost::mutex _mutex_lock_series;
    boost::mutex _mutex;
    std::string _locked_series;
private:
    DISALLOW_COPY_AND_ASSIGN(DBAIDispatcher);
};

MED_IMG_END_NAMESPACE

#endif