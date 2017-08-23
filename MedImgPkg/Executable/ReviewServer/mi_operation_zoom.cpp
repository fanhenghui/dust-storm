#include "mi_operation_zoom.h"

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

OpZoom::OpZoom() {}

OpZoom::~OpZoom() {}

int OpZoom::execute() {
  const unsigned int cell_id = _header._cell_id;
  REVIEW_CHECK_NULL_EXCEPTION(_buffer);

  MsgMouse msg;
  if (!msg.ParseFromArray(_buffer, _header._data_len)) {
    REVIEW_THROW_EXCEPTION("parse mouse message failed!");
  }

  const float pre_x = msg.pre().x();
  const float pre_y = msg.pre().y();
  const float cur_x = msg.cur().x();
  const float cur_y = msg.cur().y();

  std::shared_ptr<AppController> controller = _controller.lock();
  REVIEW_CHECK_NULL_EXCEPTION(controller);

  std::shared_ptr<AppCell> cell = controller->get_cell(cell_id);
  REVIEW_CHECK_NULL_EXCEPTION(cell);

  std::shared_ptr<SceneBase> scene = cell->get_scene();
  REVIEW_CHECK_NULL_EXCEPTION(scene);

  scene->zoom(Point2(pre_x, pre_y), Point2(cur_x, cur_y));
  std::cout << "pre pos : " << pre_x << " " << pre_y << "  ";
  std::cout << "cur pos : " << cur_x << " " << cur_y << std::endl;
  return 0;
}

MED_IMG_END_NAMESPACE