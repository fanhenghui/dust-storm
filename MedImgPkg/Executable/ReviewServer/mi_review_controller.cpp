#include "mi_review_controller.h"

#include "MedImgRenderAlgorithm/mi_volume_infos.h"

#include "MedImgAppCommon/mi_operation_command_handler.h"
#include "MedImgAppCommon/mi_ready_command_handler.h"
#include "MedImgAppCommon/mi_shut_down_command_handler.h"
#include "MedImgAppCommon/mi_app_common_define.h"
#include "mi_load_series_command_handler.h"

#include "MedImgUtil/mi_configuration.h"

MED_IMG_BEGIN_NAMESPACE

ReviewController::ReviewController()
{

}

ReviewController::~ReviewController()
{

}

void ReviewController::initialize()
{
    Configuration::instance()->set_processing_unit_type(GPU);

    std::shared_ptr<AppController> app_controller = shared_from_this();
    std::shared_ptr<ReadyCommandHandler> handler_ready(new ReadyCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_READY , handler_ready);

    std::shared_ptr<ShutDownCommandHandler> handler_shutdown(new ShutDownCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_SHUT_DOWN , handler_shutdown);

    std::shared_ptr<OperationCommandHandler> handler_operation(new OperationCommandHandler(app_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_OPERATION , handler_operation);

    std::shared_ptr<ReviewController> review_controller = std::dynamic_pointer_cast<ReviewController>(app_controller);
    std::shared_ptr<LoadSeriesCommandHandler> handler_loadseries(new LoadSeriesCommandHandler(review_controller));
    _proxy->register_command_handler(COMMAND_ID_FE_LOAD_SERIES , handler_loadseries);
}

void ReviewController::set_volume_infos(std::shared_ptr<VolumeInfos> volumeinfos)
{
    _volumeinfos = volumeinfos;
}

std::shared_ptr<VolumeInfos> ReviewController::get_volume_infos()
{
    return _volumeinfos;
}




MED_IMG_END_NAMESPACE