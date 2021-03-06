#include "mi_be_cmd_handler_db_send_preprocess_mask.h"

#include "util/mi_memory_shield.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_file_util.h"

#include "arithmetic/mi_run_length_operator.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_message.pb.h"

#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mask_label_store.h"

#include "mi_app_controller.h"
#include "mi_app_thread_model.h"
#include "mi_app_common_logger.h"
#include "mi_app_common_util.h"
#include "mi_model_dbs_status.h"

MED_IMG_BEGIN_NAMESPACE

BECmdHandlerDBSendPreprocessMask::BECmdHandlerDBSendPreprocessMask(std::shared_ptr<AppController> controller):
    _controller(controller) {

}

BECmdHandlerDBSendPreprocessMask::~BECmdHandlerDBSendPreprocessMask() {

}

int BECmdHandlerDBSendPreprocessMask::handle_command(const IPCDataHeader& ipcheader , char* buffer) {
    MI_APPCOMMON_LOG(MI_TRACE) << "IN BECmdHandlerDBSendPreprocessMask.";

    MemShield shield(buffer);
    std::shared_ptr<AppController> controller = _controller.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(controller);

    std::shared_ptr<ModelDBSStatus> model_dbs_status = AppCommonUtil::get_model_dbs_status(controller);
    APPCOMMON_CHECK_NULL_EXCEPTION(model_dbs_status);

    //get mask
    std::shared_ptr<VolumeInfos> volume_infos = controller->get_volume_infos();
    if (nullptr == volume_infos) {
        model_dbs_status->push_error_info("volume info is null when recv dbs preprocess mask");
        return -1;
    }
    std::shared_ptr<ImageData> mask = volume_infos->get_mask();
    if (nullptr == mask) {
        model_dbs_status->push_error_info("mask is null when recv dbs preprocess mask");
        return -1;
    }

    if (nullptr == buffer) {
        model_dbs_status->push_error_info("IPC buffer is null when recv dbs preprocess mask");
        return -1;
    }

    const unsigned int volume_size = mask->get_data_size();
    if(0 == RunLengthOperator::decode((unsigned int*)buffer,ipcheader.data_len/sizeof(unsigned int), (unsigned char*)mask->get_pixel_pointer(), volume_size)) {
        MaskLabelStore::instance()->fill_label(1);
        volume_infos->cache_original_mask();
        //debug
        //FileUtil::write_raw("/home/wangrui22/data/mask.raw", (unsigned char*)mask->get_pixel_pointer(), volume_size);
        MI_APPCOMMON_LOG(MI_INFO) << "decode DBS compressed preprocess mask success.";
    } else {
        model_dbs_status->push_error_info("decode IPC buffer failed when recv dbs preprocess mask");
        return -1; 
    }

    //set preprocess mask tag
    model_dbs_status->set_preprocess_mask();

    MI_APPCOMMON_LOG(MI_TRACE) << "OUT BECmdHandlerDBSendPreprocessMask.";
    return 0;
}

MED_IMG_END_NAMESPACE
