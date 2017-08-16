#include "mi_load_series_command_handler.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgUtil/mi_file_util.h"

#include "MedImgIO/mi_pacs_communicator.h"
#include "MedImgIO/mi_worklist_info.h"

#include "MedImgGLResource/mi_gl_context.h"

#include "MedImgRenderAlgorithm/mi_mpr_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_vr_scene.h"

#include "MedImgRenderAlgorithm/mi_color_transfer_function.h"
#include "MedImgRenderAlgorithm/mi_opacity_transfer_function.h"
#include "MedImgRenderAlgorithm/mi_transfer_function_loader.h"

#include "MedImgAppCommon//mi_app_thread_model.h"
#include "MedImgAppCommon/mi_app_cell.h"
#include "MedImgAppCommon/mi_app_common_define.h"

#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_review_config.h"
#include "mi_review_controller.h"

MED_IMG_BEGIN_NAMESPACE

LoadSeriesCommandHandler::LoadSeriesCommandHandler(
    std::shared_ptr<ReviewController> controller)
    : _controller(controller), _pacs_communicator(new PACSCommunicator()) {}

LoadSeriesCommandHandler::~LoadSeriesCommandHandler() {}

int LoadSeriesCommandHandler::handle_command(const IPCDataHeader &ipcheader,
                                             char *buffer) {
  std::cout << "IN load serise\n";
  // Test
  //目前只能load一次
  static bool load = false;
  if (load) {
    return 0;
  }

  std::shared_ptr<ReviewController> controller = _controller.lock();
  if (nullptr == controller) {
    REVIEW_THROW_EXCEPTION("controller pointer is null!");
  }

  std::shared_ptr<AppThreadModel> thread_model = controller->get_thread_model();
  REVIEW_CHECK_NULL_EXCEPTION(thread_model);

  std::shared_ptr<GLContext> gl_context = thread_model->get_gl_context();

  gl_context->make_current(MAIN_CONTEXT);

  CHECK_GL_ERROR;

  const unsigned int cell_id = ipcheader._msg_info0;
  const unsigned int op_id = ipcheader._msg_info1;

// 1 load series
//#define LOAD_LOCAL
//#define LOAD_PACS

#ifdef LOAD_LOCAL
  const std::string series_path(ReviewConfig::instance()->get_test_data_root() +
                                "/AB_CTA_01/");
#else
  if (!_pacs_communicator->initialize("../Config/pacs_config.txt")) {
    std::cout << "connect PACS failed!\n";
  }
  if (!_pacs_communicator->populate_whole_work_list()) {
    std::cout << "get work list failed!\n";
  }
  const std::vector<WorkListInfo> &ls = _pacs_communicator->get_work_list();
  std::cout << "worklist : \n";
  for (auto it = ls.begin(); it != ls.end(); ++it) {
    std::cout << it->GetStudyInsUID() << "   " << it->GetSeriesInsUID()
              << std::endl;
  }
  std::cout << "choose first one : " << ls[3].GetSeriesInsUID();
  const std::string series_path = _pacs_communicator->fetch_dicom(ls[3]);
  std::cout << "path is : " << series_path << std::endl;

#endif

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

  // 2 construct volume infos
  std::shared_ptr<VolumeInfos> volume_infos(new VolumeInfos());
  volume_infos->set_data_header(data_header);
  // SharedWidget::instance()->makeCurrent();
  volume_infos->set_volume(img_data); // load volume texture if has graphic card

  // Create empty mask
  std::shared_ptr<ImageData> mask_data(new ImageData());
  img_data->shallow_copy(mask_data.get());
  mask_data->_channel_num = 1;
  mask_data->_data_type = medical_imaging::UCHAR;
  mask_data->mem_allocate();
  volume_infos->set_mask(mask_data);

  controller->set_volume_infos(volume_infos);

  // 3 construct cell
  std::shared_ptr<AppCell> cell(new AppCell);

#ifdef MPR

  std::shared_ptr<MPRScene> mpr_scene(new MPRScene(512, 512));
  cell->set_scene(mpr_scene);

  const float PRESET_CT_LUNGS_WW = 1500;
  const float PRESET_CT_LUNGS_WL = -400;

  mpr_scene->set_volume_infos(volume_infos);
  mpr_scene->set_sample_rate(1.0);
  mpr_scene->set_global_window_level(PRESET_CT_LUNGS_WW, PRESET_CT_LUNGS_WL);
  mpr_scene->set_composite_mode(COMPOSITE_AVERAGE);
  mpr_scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
  mpr_scene->set_mask_mode(MASK_NONE);
  mpr_scene->set_interpolation_mode(LINEAR);
  mpr_scene->place_mpr(TRANSVERSE);

#else

  std::shared_ptr<VRScene> vr_scene(new VRScene(512, 512));
  cell->set_scene(vr_scene);

  const float PRESET_CT_LUNGS_WW = 1500;
  const float PRESET_CT_LUNGS_WL = -400;

  vr_scene->set_volume_infos(volume_infos);
  vr_scene->set_sample_rate(1.0);
  vr_scene->set_global_window_level(PRESET_CT_LUNGS_WW, PRESET_CT_LUNGS_WL);
  vr_scene->set_composite_mode(COMPOSITE_DVR);
  vr_scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
  vr_scene->set_mask_mode(MASK_NONE);
  vr_scene->set_interpolation_mode(LINEAR);
  vr_scene->set_shading_mode(SHADING_PHONG);
  vr_scene->set_proxy_geometry(PG_BRICKS); 

  // load color opacity
  const std::string color_opacity_xml =
      "/home/wr/program/git/dust-storm/MedImgPkg/Config/lut/3d/ct_cta.xml";

  std::shared_ptr<ColorTransFunc> color;
  std::shared_ptr<OpacityTransFunc> opacity;
  float ww, wl;
  RGBAUnit background;
  Material material;
  if (IO_SUCCESS !=
      TransferFuncLoader::load_color_opacity(color_opacity_xml, color, opacity,
                                             ww, wl, background, material)) {
    std::cout << "load color opacity failed!\n";
  }
  vr_scene->set_color_opacity(color, opacity, 0);
  vr_scene->set_ambient_color(1.0f, 1.0f, 1.0f, 0.28f);
  vr_scene->set_material(material, 0);
  vr_scene->set_window_level(ww, wl, 0);
  vr_scene->set_test_code(0);

#endif
  controller->add_cell(0, cell);

  CHECK_GL_ERROR;

  gl_context->make_noncurrent();

  std::cout << "load series done\n";
  load = true;
  return 0;
}

MED_IMG_END_NAMESPACE
