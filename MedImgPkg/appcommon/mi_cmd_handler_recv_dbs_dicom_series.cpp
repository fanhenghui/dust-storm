#include "mi_cmd_handler_recv_dbs_dicom_series.h"

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_client_proxy.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_logger.h"
#include "mi_message.pb.h"


MED_IMG_BEGIN_NAMESPACE

CmdHandlerRecvDBSDCMSeries::CmdHandlerRecvDBSDCMSeries(std::shared_ptr<AppController> controller):
    _controller(controller),_cur_slice(0),_total_slice(0),_th_running(false) {

}

CmdHandlerRecvDBSDCMSeries::~CmdHandlerRecvDBSDCMSeries() {
    _th_cache_db.join();
}

int CmdHandlerRecvDBSDCMSeries::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN recveive DB server DICOM series cmd handler.";
    std::shared_ptr<AppController> controller = _controller.lock();
    if (nullptr == controller) {
        APPCOMMON_THROW_EXCEPTION("controller pointer is null!");
    }

    const unsigned int end_tag = ipcheader.msg_info2;
    const unsigned int dicom_slice = ipcheader.msg_info3;

    //trigger a thread to write to cache DB
    if (!_th_running) {
        _th_cache_db = boost::thread(boost::bind(&CmdHandlerRecvDBSDCMSeries::update_cache_db_i, this));
        _th_running = true;
        _cur_slice = 0;
        _total_slice = dicom_slice;
    }

    //read one slice

    //get number image or sop  and push to queue

    //read rest

    std::shared_ptr<DCMSlice> ;

    //push to 
    //load one slice

    
    
    ++_cur_slice;
    MI_APPCOMMON_LOG(MI_TRACE) << "OUT recveive DB server DICOM series cmd handler.";

    return 0;
}

void CmdHandlerRecvDBSDCMSeries::update_cache_db_i() {
    while(true) {
        //pop slice buffer to write to local disk

        if (_cur_slice == _total_slice) {
            //write path to cache db

            break;
        }
    }
}


MED_IMG_END_NAMESPACE
