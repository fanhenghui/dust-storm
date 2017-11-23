#include "mi_review_controller.h"

#include "util/mi_configuration.h"

#include "renderalgo/mi_volume_infos.h"

#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_thread_model.h"
#include "appcommon/mi_cmd_handler_operating.h"
#include "appcommon/mi_operation_factory.h"
#include "appcommon/mi_cmd_handler_fe_ready.h"
#include "appcommon/mi_cmd_handler_shutdown.h"
#include "appcommon/mi_cmd_handler_heartbeat.h"
#include "appcommon/mi_cmd_handler_mpr_play.h"
#include "appcommon/mi_cmd_handler_vr_play.h"
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
#include "appcommon/mi_cmd_handler_search_worklist.h"
#include "appcommon/mi_cmd_handler_back_to_worklist.h"
#include "appcommon/mi_cmd_handler_recv_dbs_ai_annotation.h"
#include "appcommon/mi_cmd_handler_recv_dbs_dicom_series.h"
#include "appcommon/mi_cmd_handler_recv_dbs_end_signal.h"
#include "appcommon/mi_cmd_handler_recv_dbs_error.h"
#include "appcommon/mi_cmd_handler_recv_dbs_preprocess_mask.h"

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
    std::shared_ptr<CmdHandlerHeartbeat> handler_heart_beat(
        new CmdHandlerHeartbeat(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_HEARTBEAT, handler_heart_beat);

    std::shared_ptr<CmdHandlerFEReady> handler_ready(
        new CmdHandlerFEReady(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_READY, handler_ready);

    std::shared_ptr<CmdHandlerShutdown> handler_shutdown(
        new CmdHandlerShutdown(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_SHUT_DOWN, handler_shutdown);

    std::shared_ptr<CmdHandlerOperating> handler_operation(
        new CmdHandlerOperating(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_OPERATION, handler_operation);

    // std::shared_ptr<CmdHandlerMPRPlay> handler_mpr_play(
    //     new CmdHandlerMPRPlay(app_controller));
    // _proxy->register_command_handler(COMMAND_ID_FE_MPR_PLAY, handler_mpr_play);

    std::shared_ptr<CmdHandlerVRPlay> handler_vr_play(
        new CmdHandlerVRPlay(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_VR_PLAY, handler_vr_play);

    std::shared_ptr<CmdHandlerSearchWorklist> handler_search_worklist(
        new CmdHandlerSearchWorklist(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_SEARCH_WORKLIST, handler_search_worklist);

    std::shared_ptr<CmdHandlerBackToWorklist> handler_back_to_worklist(
        new CmdHandlerBackToWorklist(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_BACK_TO_WORKLIST, handler_back_to_worklist);

    // Register operation
    OperationFactory::instance()->register_operation(
        OPERATION_ID_MPR_PAGING, std::shared_ptr<OpMPRPaging>(new OpMPRPaging()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_INIT, std::shared_ptr<OpInit>(new OpInit()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_ROTATE, std::shared_ptr<OpRotate>(new OpRotate()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_ZOOM, std::shared_ptr<OpZoom>(new OpZoom()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_PAN, std::shared_ptr<OpPan>(new OpPan()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_WINDOWING, std::shared_ptr<OpWindowing>(new OpWindowing()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_RESIZE, std::shared_ptr<OpResize>(new OpResize()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_MPR_MASK_OVERLAY, std::shared_ptr<OpMPRMaskOverlay>(new OpMPRMaskOverlay()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_SWITCH_PRESET_WINDOWING, std::shared_ptr<OpSwitchPresetWindowing>(new OpSwitchPresetWindowing()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_SWITCH_PRESET_VRT, std::shared_ptr<OpSwitchPresetVRT>(new OpSwitchPresetVRT()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_LOCATE, std::shared_ptr<OpLocate>(new OpLocate()));
    OperationFactory::instance()->register_operation(
        OPERATION_ID_DOWNSAMPLE, std::shared_ptr<OpDownsample>(new OpDownsample()));    
    OperationFactory::instance()->register_operation(
        OPERATION_ID_ANNOTATION, std::shared_ptr<OpAnnotation>(new OpAnnotation()));

    //register command handler for DBS
    _client_proxy_dbs->register_command_handler(COMMAND_ID_DB_SEND_DICOM_SERIES, 
        std::shared_ptr<CmdHandlerRecvDBSDCMSeries>(new CmdHandlerRecvDBSDCMSeries(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_DB_SEND_PREPROCESS_MASK, 
    std::shared_ptr<CmdHandlerRecvDBSPreprocessMask>(new CmdHandlerRecvDBSPreprocessMask(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_DB_SEND_AI_ANNOTATION, 
    std::shared_ptr<CmdHandlerRecvDBSAIAnno>(new CmdHandlerRecvDBSAIAnno(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_DB_SEND_END, 
    std::shared_ptr<CmdHandlerRecvDBSEndSignal>(new CmdHandlerRecvDBSEndSignal(app_controller)));
    _client_proxy_dbs->register_command_handler(COMMAND_ID_DB_SEND_ERROR, 
    std::shared_ptr<CmdHandlerRecvDBSError>(new CmdHandlerRecvDBSError(app_controller)));
}


MED_IMG_END_NAMESPACE