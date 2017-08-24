#include "mi_operation_resize.h"

#include "MedImgAppCommon/mi_app_cell.h"
#include "MedImgAppCommon/mi_app_controller.h"

#include "MedImgIO/mi_image_data.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_vr_scene.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "mi_message.pb.h"
#include "mi_review_controller.h"

MED_IMG_BEGIN_NAMESPACE

OpResize::OpResize() {}

OpResize::~OpResize() {}

int OpResize::execute() {
  const unsigned int cell_id = _header._cell_id;
  REVIEW_CHECK_NULL_EXCEPTION(_buffer);

  MsgResize msg;
  if (!msg.ParseFromArray(_buffer, _header._data_len)) {
    REVIEW_THROW_EXCEPTION("parse resize message failed!");
  }

  std::shared_ptr<AppController> controller = _controller.lock();
  REVIEW_CHECK_NULL_EXCEPTION(controller);
  for (size_t i = 0; i < msg.cells_size(); i++) {
    const MsgCellInfo &cell_info = msg.cells(i);
    const int cell_id = cell_info.id();
    const int width = cell_info.width();
    const int height = cell_info.height();

    std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
    REVIEW_CHECK_NULL_EXCEPTION(cell);
    std::shared_ptr<SceneBase> scene = cell->get_scene();
    scene->set_display_size(width, height);

    printf("resize cellid % d , width %d , height %d \n", cell_id, width,
           height);
  }

  return 0;
}

MED_IMG_END_NAMESPACE