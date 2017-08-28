#include "mi_operation_init.h"
#include "mi_message.pb.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_client_proxy.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"

#include "io/mi_pacs_communicator.h"
#include "io/mi_worklist_info.h"

#include "glresource/mi_gl_context.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"

#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_opacity_transfer_function.h"
#include "renderalgo/mi_transfer_function_loader.h"

#include "MedImgAppCommon//mi_app_thread_model.h"
#include "MedImgAppCommon/mi_app_cell.h"
#include "MedImgAppCommon/mi_app_common_define.h"
#include "MedImgAppCommon/mi_app_data_base.h"

#include "glresource/mi_gl_utils.h"

#include "mi_review_config.h"
#include "mi_review_controller.h"

MED_IMG_BEGIN_NAMESPACE

OpInit::OpInit() {}

OpInit::~OpInit() {}

int OpInit::execute() {
  std::shared_ptr<AppController> app_controller(_controller.lock());
  std::shared_ptr<ReviewController> controller =
      std::dynamic_pointer_cast<ReviewController>(app_controller);
  REVIEW_CHECK_NULL_EXCEPTION(controller);

  REVIEW_CHECK_NULL_EXCEPTION(_buffer);
  MsgInit msg_init;
  if (!msg_init.ParseFromArray(_buffer, _header._data_len)) {
    // TODO ERROR
    std::cout << "parse message failed!\n";
    return -1;
  }

  // std::shared_ptr<AppThreadModel> thread_model =
  // controller->get_thread_model();
  // REVIEW_CHECK_NULL_EXCEPTION(thread_model);
  // std::shared_ptr<GLContext> gl_context = thread_model->get_gl_context();
  // gl_context->make_current(MAIN_CONTEXT);

  // load series
  const std::string series_uid = msg_init.series_uid();
  AppDataBase db;
  if(0 != db.connect("root","127.0.0.1:3306","6ckj1sWR","med_img_cache_db")){
      //TODO LOG
      return -1;
  }

  std::string data_path;
  if(0 != db.get_series_path(series_uid , data_path) ){
      //TODO LOG
      return -1;
  }
  const std::string series_path(data_path);

  ////////////////////////////////////////////////////////////////////////
  // TODO TEST
  // const std::string series_path(ReviewConfig::instance()->get_test_data_root() +
  //                               "/AB_CTA_01/");
  ////////////////////////////////////////////////////////////////////////

  std::vector<std::string> dcm_files;
  FileUtil::get_all_file_recursion(series_path, std::vector<std::string>(),
                                   dcm_files);
  if (dcm_files.empty()) {
    REVIEW_THROW_EXCEPTION("Empty series files!");
  }
  std::shared_ptr<ImageDataHeader> data_header;
  std::shared_ptr<ImageData> img_data;
  DICOMLoader loader;
  IOStatus status = loader.load_series(dcm_files, img_data, data_header);
  if (status != IO_SUCCESS) {
    REVIEW_THROW_EXCEPTION("load series failed");
  }

  // create volume infos
  std::shared_ptr<VolumeInfos> volume_infos(new VolumeInfos());
  volume_infos->set_data_header(data_header);
  volume_infos->set_volume(img_data); // load volume texture if has graphic card

  // create empty mask
  std::shared_ptr<ImageData> mask_data(new ImageData());
  img_data->shallow_copy(mask_data.get());
  mask_data->_channel_num = 1;
  mask_data->_data_type = medical_imaging::UCHAR;
  mask_data->mem_allocate();
  volume_infos->set_mask(mask_data);
  controller->set_volume_infos(volume_infos);

  // create cells
  for (int i = 0; i < msg_init.cells_size(); ++i) {
    const MsgCellInfo &cell_info = msg_init.cells(i);
    const int width = cell_info.width();
    const int height = cell_info.height();
    const int direction = cell_info.direction();
    const int cell_id = cell_info.id();
    const int type_id = cell_info.type();

    const float DEFAULT_WW = 1500;
    const float DEFAULT_WL = -400;

    std::shared_ptr<AppCell> cell(new AppCell);
    if (type_id == 1) { // MPR
      std::shared_ptr<MPRScene> mpr_scene(new MPRScene(width, height));
      cell->set_scene(mpr_scene);
      mpr_scene->set_test_code(0);
      {
        std::stringstream ss;
        ss << "cell_" << cell_id;
        mpr_scene->set_name(ss.str());
      }

      mpr_scene->set_volume_infos(volume_infos);
      mpr_scene->set_sample_rate(1.0);
      mpr_scene->set_global_window_level(DEFAULT_WW, DEFAULT_WL);
      mpr_scene->set_composite_mode(COMPOSITE_AVERAGE);
      mpr_scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
      mpr_scene->set_mask_mode(MASK_NONE);
      mpr_scene->set_interpolation_mode(LINEAR);
      switch (direction) {
      case 0:
        mpr_scene->place_mpr(SAGITTAL);
  std::cout << "out init op\n";
  break;
      case 1:
        mpr_scene->place_mpr(CORONAL);
        break;
      default: // 2
        mpr_scene->place_mpr(TRANSVERSE);
        break;
      }

    } else if (type_id == 2) { // VR
      std::shared_ptr<VRScene> vr_scene(new VRScene(width, height));
      cell->set_scene(vr_scene);
      vr_scene->set_test_code(0);
      {
        std::stringstream ss;
        ss << "cell_" << cell_id;
        vr_scene->set_name(ss.str());
      }
      vr_scene->set_volume_infos(volume_infos);
      vr_scene->set_sample_rate(1.0);
      vr_scene->set_global_window_level(DEFAULT_WW, DEFAULT_WL);
      vr_scene->set_composite_mode(COMPOSITE_DVR);
      vr_scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
      vr_scene->set_mask_mode(MASK_NONE);
      vr_scene->set_interpolation_mode(LINEAR);
      vr_scene->set_shading_mode(SHADING_PHONG);
      vr_scene->set_proxy_geometry(PG_BRICKS);

      // load color opacity
      const std::string color_opacity_xml =
          "../Config/lut/3d/ct_cta.xml";
      std::shared_ptr<ColorTransFunc> color;
      std::shared_ptr<OpacityTransFunc> opacity;
      float ww, wl;
      RGBAUnit background;
      Material material;
      if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(
                            color_opacity_xml, color, opacity, ww, wl,
                            background, material)) {
        // TODO ERROR
        REVIEW_THROW_EXCEPTION("load color&opacity failed");
      }
      vr_scene->set_color_opacity(color, opacity, 0);
      vr_scene->set_ambient_color(1.0f, 1.0f, 1.0f, 0.28f);
      vr_scene->set_material(material, 0);
      vr_scene->set_window_level(ww, wl, 0);
      vr_scene->set_test_code(0);
    } else {
      REVIEW_THROW_EXCEPTION("invalid cell id!");
    }

    controller->add_cell(cell_id, cell);
  }

  return 0;
}

MED_IMG_END_NAMESPACE