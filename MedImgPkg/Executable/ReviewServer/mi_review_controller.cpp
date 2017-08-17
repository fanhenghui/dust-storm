#include "mi_review_controller.h"

#include "MedImgRenderAlgorithm/mi_volume_infos.h"

#include "MedImgAppCommon/mi_app_common_define.h"
#include "MedImgAppCommon/mi_app_thread_model.h"
#include "MedImgAppCommon/mi_operation_command_handler.h"
#include "MedImgAppCommon/mi_operation_factory.h"
#include "MedImgAppCommon/mi_ready_command_handler.h"
#include "MedImgAppCommon/mi_shut_down_command_handler.h"

#include "MedImgUtil/mi_configuration.h"

#include "mi_load_series_command_handler.h"
#include "mi_mpr_play_command_handler.h"
#include "mi_operation_mpr_paging.h"
#include "mi_operation_init.h"

MED_IMG_BEGIN_NAMESPACE

ReviewController::ReviewController() {}

ReviewController::~ReviewController() {}

void ReviewController::initialize() {
  AppController::initialize();

  // Configure
  Configuration::instance()->set_processing_unit_type(GPU);

  // Register command handler
  std::shared_ptr<AppController> app_controller = shared_from_this();
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

  //   std::shared_ptr<ReviewController> review_controller =
  //       std::dynamic_pointer_cast<ReviewController>(app_controller);
  //   std::shared_ptr<LoadSeriesCommandHandler> handler_loadseries(
  //       new LoadSeriesCommandHandler(review_controller));
  //   _proxy->register_command_handler(COMMAND_ID_FE_LOAD_SERIES,
  //                                    handler_loadseries);

  // Register operation
  OperationFactory::instance()->register_operation(
      OPERATION_ID_MPR_PAGING, std::shared_ptr<OpMPRPaging>(new OpMPRPaging()));
  OperationFactory::instance()->register_operation(
      OPERATION_ID_INIT, std::shared_ptr<OpInit>(new OpInit()));
}

void ReviewController::set_volume_infos(
    std::shared_ptr<VolumeInfos> volumeinfos) {
  _volumeinfos = volumeinfos;
}

std::shared_ptr<VolumeInfos> ReviewController::get_volume_infos() {
  return _volumeinfos;
}

MED_IMG_END_NAMESPACE