#ifndef MED_IMG_APPCOMMON_MI_CMD_HANDLER_RECV_DB_SERVER_AI_ANNOTATION_H
#define MED_IMG_APPCOMMON_MI_CMD_HANDLER_RECV_DB_SERVER_AI_ANNOTATION_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>
#include <deque>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

MED_IMG_BEGIN_NAMESPACE

class AppController;
class AppCommon_Export CmdHandlerRecvDBSDCMSeries : public ICommandHandler {
public:
    CmdHandlerRecvDBSDCMSeries(std::shared_ptr<AppController> controller);

    virtual ~CmdHandlerRecvDBSDCMSeries();

    virtual int handle_command(const IPCDataHeader& dataheader , char* buffer);

private:
    void update_cache_db_i();
private:
    std::weak_ptr<AppController> _controller;

    bool _th_running;
    boost::condition _condition_cache_db;
    boost::thread _th_cache_db;
    boost::mutex _mutex_cache_db;

    struct DCMSlice {
        char* buffer;
        int image_number;//00200013
        
        DCMSlice(char* buffer_, int image_number_):
            buffer(buffer_),image_number(image_number_) {}
        ~DCMSlice() {
            if (nullptr != buffer) {
                delete [] buffer;
                buffer = nullptr;
            }
        }

    };
    std::deque<std::shared_ptr<DCMSlice>> _dcm_queue;
    std::string _dcm_root;
    int _cur_slice;
    int _total_slice;
};

MED_IMG_END_NAMESPACE


#endif