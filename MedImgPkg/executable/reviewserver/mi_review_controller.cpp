#include "mi_review_controller.h"

#include "util/mi_configuration.h"

#include "renderalgo/mi_volume_infos.h"

#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_thread_model.h"
#include "appcommon/mi_cmd_handler_be_fe_operation.h"
#include "appcommon/mi_operation_factory.h"
#include "appcommon/mi_cmd_handler_be_fe_ready.h"
#include "appcommon/mi_cmd_handler_be_fe_shutdown.h"
#include "appcommon/mi_cmd_handler_be_fe_heartbeat.h"
#include "appcommon/mi_cmd_handler_be_fe_play_vr.h"
#include "appcommon/mi_operation_mpr_paging.h"
#include "appcommon/mi_operation_pan.h"
#include "appcommon/mi_operation_resize.h"
#include "appcommon/mi_operation_rotate.h"
#include "appcommon/mi_operation_windowing.h"
#include "appcommon/mi_operation_zoom.h"
#include "appcommon/mi_operation_annotation.h"
#include "appcommon/mi_operation_mpr_mask_overlay.h"
#include "appcommon/mi_operation_switch_preset_windowing.h"
#include "appcommon/mi_operation_switch_preset_vrt.h"
#include "appcommon/mi_operation_locate.h"
#include "appcommon/mi_operation_downsample.h"
#include "appcommon/mi_model_annotation.h"
#include "appcommon/mi_model_crosshair.h"
#include "appcommon/mi_cmd_handler_be_fe_db_retrieve.h"
#include "appcommon/mi_cmd_handler_be_fe_back_to_worklist.h"
#include "appcommon/mi_cmd_handler_be_db_send_ai_evaluation.h"
#include "appcommon/mi_cmd_handler_be_db_send_dicom.h"
#include "appcommon/mi_cmd_handler_be_db_send_end_signal.h"
#include "appcommon/mi_cmd_handler_be_db_send_error.h"
#include "appcommon/mi_cmd_handler_be_db_send_preprocess_mask.h"
#include "appcommon/mi_cmd_handler_be_fe_pacs_retrieve.h"
#include "appcommon/mi_cmd_handler_be_fe_pacs_fetch.h"
#include "appcommon/mi_cmd_handler_be_db_pacs_retrieve_result.h"
#include "appcommon/mi_cmd_handler_be_db_pacs_fetch_result.h"

#include "mi_operation_init.h"

MED_IMG_BEGIN_NAMESPACE

ReviewController::ReviewController() {}

ReviewController::~ReviewController() {}

void ReviewController::initialize() {
    AppController::initialize();
    // configure
    Configuration::instance()->set_processing_unit_type(GPU);
    // register command handler and operation
    register_command_handler_i();
}

void ReviewController::register_command_handler_i() {
    // Register command handler
    std::shared_ptr<AppController> app_controller = shared_from_this();
    ;
    _proxy->register_command_handler(COMMAND_ID_BE_FE_HEARTBEAT, 
    std::shared_ptr<CmdHandlerBE_FEHeartbeat>(new CmdHandlerBE_FEHeartbeat(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_READY, 
    std::shared_ptr<CmdHandlerBE_FEReady>(new CmdHandlerBE_FEReady(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_SHUTDOWN,
    std::shared_ptr<CmdHandlerBE_FEShutdown>(new CmdHandlerBE_FEShutdown(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_OPERATION, 
    std::shared_ptr<CmdHandlerBE_FEOperation>(new CmdHandlerBE_FEOperation(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_PLAY_VR,
    std::shared_ptr<CmdHandlerBE_FEPlayVR>(new CmdHandlerBE_FEPlayVR(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_DB_RETRIEVE,
    std::shared_ptr<CmdHandlerBE_FEDBRetrieve>(new CmdHandlerBE_FEDBRetrieve(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_BACK_TO_WORKLIST, 
    std::shared_ptr<CmdHandlerBE_FEBackToWorklist> (new CmdHandlerBE_FEBackToWorklist(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_PACS_RETRIEVE, 
    std::shared_ptr<CmdHandlerBE_FEPACSRetrieve>(new CmdHandlerBE_FEPACSRetrieve(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_PACS_FETCH, 
    std::shared_ptr<CmdHandlerBE_FEPACSFetch>(new CmdHandlerBE_FEPACSFetch(app_controller)));



    // Register operation
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_MPR_PAGING, std::shared_ptr<OpMPRPaging>(new OpMPRPaging()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_INIT, std::shared_ptr<OpInit>(new OpInit()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_ROTATE, std::shared_ptr<OpRotate>(new OpRotate()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_ZOOM, std::shared_ptr<OpZoom>(new OpZoom()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_PAN, std::shared_ptr<OpPan>(new OpPan()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_WINDOWING, std::shared_ptr<OpWindowing>(new OpWindowing()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_RESIZE, std::shared_ptr<OpResize>(new OpResize()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_MPR_MASK_OVERLAY, std::shared_ptr<OpMPRMaskOverlay>(new OpMPRMaskOverlay()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_SWITCH_PRESET_WINDOWING, std::shared_ptr<OpSwitchPresetWindowing>(new OpSwitchPresetWindowing()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_SWITCH_PRESET_VRT, std::shared_ptr<OpSwitchPresetVRT>(new OpSwitchPresetVRT()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_LOCATE, std::shared_ptr<OpLocate>(new OpLocate()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_DOWNSAMPLE, std::shared_ptr<OpDownsample>(new OpDownsample()));    
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_ANNOTATION, std::shared_ptr<OpAnnotation>(new OpAnnotation()));

    //register command handler for DBS
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_DICOM_SERIES, 
        std::shared_ptr<CmdHandlerBE_DBSendDICOM>(new CmdHandlerBE_DBSendDICOM(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_PREPROCESS_MASK, 
    std::shared_ptr<CmdHandlerBE_DBSendPreprocessMask>(new CmdHandlerBE_DBSendPreprocessMask(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_AI_EVALUATION, 
    std::shared_ptr<CmdHandlerBE_DBSendAIEvaluation>(new CmdHandlerBE_DBSendAIEvaluation(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_END, 
    std::shared_ptr<CmdHandlerBE_DBSendEndSignal>(new CmdHandlerBE_DBSendEndSignal(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_ERROR, 
    std::shared_ptr<CmdHandlerBE_DBSendError>(new CmdHandlerBE_DBSendError(app_controller)));

    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_PACS_RETRIEVE_RESULT, 
    std::shared_ptr<CmdHandlerBE_DBPACSRetrieveResult>(new CmdHandlerBE_DBPACSRetrieveResult(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_PACS_FETCH_RESULT, 
    std::shared_ptr<CmdHandlerBE_DBPACSFetchResult>(new CmdHandlerBE_DBPACSFetchResult(app_controller)));
}


MED_IMG_END_NAMESPACE