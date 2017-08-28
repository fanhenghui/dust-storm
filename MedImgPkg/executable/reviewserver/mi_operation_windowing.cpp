#include "mi_operation_windowing.h"

#include "appcommon/mi_app_cell.h"
#include "appcommon/mi_app_controller.h"

#include "io/mi_image_data.h"
#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"

#include "arithmetic/mi_ortho_camera.h"

#include "mi_message.pb.h"
#include "mi_review_controller.h"

MED_IMG_BEGIN_NAMESPACE 

OpWindowing::OpWindowing() {}

OpWindowing::~OpWindowing() {}

int OpWindowing::execute() {
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

  std::shared_ptr<VRScene> scene_ext =
      std::dynamic_pointer_cast<VRScene>(scene);
  if (scene_ext) {
    float ww, wl;
    if (scene_ext->get_composite_mode() == COMPOSITE_DVR) {
      // TODO consider when has mask
      scene_ext->get_window_level(ww, wl, 0);
      ww += (cur_x - pre_x);
      wl += (pre_y - cur_y);
      scene_ext->set_window_level(ww, wl, 0);
    } else {
      scene_ext->get_global_window_level(ww, wl);
      ww += (cur_x - pre_x);
      wl += (pre_y - cur_y);
      scene_ext->set_global_window_level(ww, wl);
    }
  } else {
    std::shared_ptr<MPRScene> scene_ext =
        std::dynamic_pointer_cast<MPRScene>(scene);
    float ww, wl;
    scene_ext->get_global_window_level(ww, wl);
    ww += (cur_x - pre_x);
    wl += (pre_y - cur_y);
    scene_ext->set_global_window_level(ww, wl);
  }

  std::cout << "pre pos : " << pre_x << " " << pre_y << "  ";
  std::cout << "cur pos : " << cur_x << " " << cur_y << std::endl;
  return 0;
}

MED_IMG_END_NAMESPACE