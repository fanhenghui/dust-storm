#include "mi_review_controller.h"

#include "util/mi_operation_factory.h"

#include "io/mi_configure.h"

#include "renderalgo/mi_volume_infos.h"

#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_thread_model.h"
#include "appcommon/mi_be_cmd_handler_fe_operation.h"
#include "appcommon/mi_be_cmd_handler_fe_ready.h"
#include "appcommon/mi_be_cmd_handler_fe_shutdown.h"
#include "appcommon/mi_be_cmd_handler_fe_heartbeat.h"
#include "appcommon/mi_be_cmd_handler_fe_play_vr.h"
#include "appcommon/mi_be_operation_fe_mpr_paging.h"
#include "appcommon/mi_be_operation_fe_pan.h"
#include "appcommon/mi_be_operation_fe_resize.h"
#include "appcommon/mi_be_operation_fe_rotate.h"
#include "appcommon/mi_be_operation_fe_windowing.h"
#include "appcommon/mi_be_operation_fe_zoom.h"
#include "appcommon/mi_be_operation_fe_annotation.h"
#include "appcommon/mi_be_operation_fe_mpr_mask_overlay.h"
#include "appcommon/mi_be_operation_fe_switch_preset_windowing.h"
#include "appcommon/mi_be_operation_fe_switch_preset_vrt.h"
#include "appcommon/mi_be_operation_fe_locate.h"
#include "appcommon/mi_be_operation_fe_downsample.h"
#include "appcommon/mi_be_operation_fe_adjust_evaluation_probability.h"
#include "appcommon/mi_model_annotation.h"
#include "appcommon/mi_model_crosshair.h"
#include "appcommon/mi_be_cmd_handler_fe_db_retrieve.h"
#include "appcommon/mi_be_cmd_handler_fe_back_to_worklist.h"
#include "appcommon/mi_be_cmd_handler_db_send_ai_evaluation.h"
#include "appcommon/mi_be_cmd_handler_db_send_dicom.h"
#include "appcommon/mi_be_cmd_handler_db_send_end_signal.h"
#include "appcommon/mi_be_cmd_handler_db_send_error.h"
#include "appcommon/mi_be_cmd_handler_db_send_preprocess_mask.h"
#include "appcommon/mi_be_cmd_handler_fe_pacs_retrieve.h"
#include "appcommon/mi_be_cmd_handler_fe_pacs_fetch.h"
#include "appcommon/mi_be_cmd_handler_db_pacs_retrieve_result.h"
#include "appcommon/mi_be_cmd_handler_db_pacs_fetch_result.h"

#include "appcommon/mi_model_pacs.h"

#include "mi_operation_init.h"

MED_IMG_BEGIN_NAMESPACE

ReviewController::ReviewController() {}

ReviewController::~ReviewController() {}

void ReviewController::initialize() {
    AppController::initialize();
    // configure
    Configure::instance()->set_processing_unit_type(GPU);
    // init default model
    init_default_model_i();
    // register command handler and operation
    register_command_handler_i();
}

void ReviewController::register_command_handler_i() {
    // Register command handler
    std::shared_ptr<AppController> app_controller = shared_from_this();
    ;
    _proxy->register_command_handler(COMMAND_ID_BE_FE_HEARTBEAT, 
    std::shared_ptr<BECmdHandlerFEHeartbeat>(new BECmdHandlerFEHeartbeat(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_READY, 
    std::shared_ptr<BECmdHandlerFEReady>(new BECmdHandlerFEReady(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_SHUTDOWN,
    std::shared_ptr<BECmdHandlerFEShutdown>(new BECmdHandlerFEShutdown(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_OPERATION, 
    std::shared_ptr<BECmdHandlerFEOperation>(new BECmdHandlerFEOperation(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_PLAY_VR,
    std::shared_ptr<BECmdHandlerFEPlayVR>(new BECmdHandlerFEPlayVR(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_DB_RETRIEVE,
    std::shared_ptr<BECmdHandlerFEDBRetrieve>(new BECmdHandlerFEDBRetrieve(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_BACK_TO_WORKLIST, 
    std::shared_ptr<BECmdHandlerFEBackToWorklist> (new BECmdHandlerFEBackToWorklist(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_PACS_RETRIEVE, 
    std::shared_ptr<BECmdHandlerFEPACSRetrieve>(new BECmdHandlerFEPACSRetrieve(app_controller)));

    _proxy->register_command_handler(COMMAND_ID_BE_FE_PACS_FETCH, 
    std::shared_ptr<BECmdHandlerFEPACSFetch>(new BECmdHandlerFEPACSFetch(app_controller)));


    // Register operation
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_MPR_PAGING, std::shared_ptr<BEOpFEMPRPaging>(new BEOpFEMPRPaging()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_INIT, std::shared_ptr<OpInit>(new OpInit()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_ROTATE, std::shared_ptr<BEOpFERotate>(new BEOpFERotate()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_ZOOM, std::shared_ptr<BEOpFEZoom>(new BEOpFEZoom()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_PAN, std::shared_ptr<BEOpFEPan>(new BEOpFEPan()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_WINDOWING, std::shared_ptr<BEOpFEWindowing>(new BEOpFEWindowing()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_RESIZE, std::shared_ptr<BEOpFEResize>(new BEOpFEResize()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_MPR_MASK_OVERLAY, std::shared_ptr<BEOpFEMPRMaskOverlay>(new BEOpFEMPRMaskOverlay()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_SWITCH_PRESET_WINDOWING, std::shared_ptr<BEOpFESwitchPresetWindowing>(new BEOpFESwitchPresetWindowing()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_SWITCH_PRESET_VRT, std::shared_ptr<BEOpFESwitchPresetVRT>(new BEOpFESwitchPresetVRT()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_LOCATE, std::shared_ptr<BEOpFELocate>(new BEOpFELocate()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_DOWNSAMPLE, std::shared_ptr<BEOpFEDownsample>(new BEOpFEDownsample()));    
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_ANNOTATION, std::shared_ptr<BEOpFEAnnotation>(new BEOpFEAnnotation()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_BE_FE_ADJUST_EVALUATION_PROBABILITY_THRESHOLD, std::shared_ptr<BEOpFEAdjustEvaluationProbability>(new BEOpFEAdjustEvaluationProbability()));

    //register command handler for DBS
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_DICOM_SERIES, 
        std::shared_ptr<BECmdHandlerDBSendDICOM>(new BECmdHandlerDBSendDICOM(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_PREPROCESS_MASK, 
    std::shared_ptr<BECmdHandlerDBSendPreprocessMask>(new BECmdHandlerDBSendPreprocessMask(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_AI_EVALUATION, 
    std::shared_ptr<BECmdHandlerDBSendAIEvaluation>(new BECmdHandlerDBSendAIEvaluation(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_END, 
    std::shared_ptr<BECmdHandlerDBSendEndSignal>(new BECmdHandlerDBSendEndSignal(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_SEND_ERROR, 
    std::shared_ptr<BECmdHandlerDBSendError>(new BECmdHandlerDBSendError(app_controller)));

    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_PACS_RETRIEVE_RESULT, 
    std::shared_ptr<BECmdHandlerDBPACSRetrieveResult>(new BECmdHandlerDBPACSRetrieveResult(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_BE_DB_PACS_FETCH_RESULT, 
    std::shared_ptr<BECmdHandlerDBPACSFetchResult>(new BECmdHandlerDBPACSFetchResult(app_controller)));
}

void ReviewController::init_default_model_i() {
    this->add_model(MODEL_ID_PACS,  std::shared_ptr<ModelPACS>(new ModelPACS()));
}

MED_IMG_END_NAMESPACE