#ifndef MED_IMG_APPCOMMON_MI_CMD_HANDLER_RECV_DB_SERVER_DICON_SERIES_H
#define MED_IMG_APPCOMMON_MI_CMD_HANDLER_RECV_DB_SERVER_DICON_SERIES_H

#include "appcommon/mi_app_common_export.h"
#include "util/mi_ipc_common.h"
#include <memory>
#include <deque>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

#include "io/mi_dicom_loader.h"

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

    std::vector<DICOMLoader::DCMSliceStream*> _dcm_streams_store;
    std::deque<DICOMLoader::DCMSliceStream*> _dcm_streams_queue;
    std::string _dcm_root;
    int _cur_recv_slice;
    int _cur_save_slice;
    int _total_slice;
    bool _err_tag;

    //Series info
    std::string _series_id;
    std::string _study_id;
    std::string _series_path;
    std::string _patient_name;
    std::string _patient_id;
    std::string _modality;
};

MED_IMG_END_NAMESPACE

#endif