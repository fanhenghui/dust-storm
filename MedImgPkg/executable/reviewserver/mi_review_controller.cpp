#include "mi_review_controller.h"

#include "renderalgo/mi_volume_infos.h"

#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_thread_model.h"
#include "appcommon/mi_operation_command_handler.h"
#include "appcommon/mi_operation_factory.h"
#include "appcommon/mi_ready_command_handler.h"
#include "appcommon/mi_shut_down_command_handler.h"
#include "appcommon/mi_heartbeat_command_handler.h"

#include "util/mi_configuration.h"

#include "mi_search_worklist_command_handler.h"
#include "mi_mpr_play_command_handler.h"
#include "mi_vr_play_command_handler.h"
#include "mi_operation_init.h"
#include "mi_operation_mpr_paging.h"
#include "mi_operation_pan.h"
#include "mi_operation_resize.h"
#include "mi_operation_rotate.h"
#include "mi_operation_windowing.h"
#include "mi_operation_zoom.h"
#include "mi_operation_annotation.h"

MED_IMG_BEGIN_NAMESPACE

ReviewController::ReviewController() {}

ReviewController::~ReviewController() {}

void ReviewController::initialize() {
    AppController::initialize();

    // Configure
    Configuration::instance()->set_processing_unit_type(GPU);

    // Register command handler
    std::shared_ptr<AppController> app_controller = shared_from_this();
    std::shared_ptr<HeartbeatCommandHandler> handler_heart_beat(
        new HeartbeatCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_HEARTBEAT, handler_heart_beat);

    std::shared_ptr<ReadyCommandHandler> handler_ready(
        new ReadyCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_READY, handler_ready);

    std::shared_ptr<ShutDownCommandHandler> handler_shutdown(
        new ShutDownCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_SHUT_DOWN, handler_shutdown);

    std::shared_ptr<OperationCommandHandler> handler_operation(
        new OperationCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_OPERATION, handler_operation);

    std::shared_ptr<MPRPlayCommandHandler> handler_mpr_play(
        new MPRPlayCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_MPR_PLAY, handler_mpr_play);

    std::shared_ptr<VRPlayCommandHandler> handler_vr_play(
        new VRPlayCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_VR_PLAY, handler_vr_play);

    std::shared_ptr<SearchWorklistCommandHandler> handler_search_worklist(
        new SearchWorklistCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_WORKLIST, handler_search_worklist);

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
        OPERATION_ID_ANNOTATION, std::shared_ptr<OpAnnotation>(new OpAnnotation()));
}

void ReviewController::set_volume_infos(
    std::shared_ptr<VolumeInfos> volumeinfos) {
    _volumeinfos = volumeinfos;
}

std::shared_ptr<VolumeInfos> ReviewController::get_volume_infos() {
    return _volumeinfos;
}

MED_IMG_END_NAMESPACE