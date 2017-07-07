#include "mi_load_series_command_handler.h"

#include "MedImgUtil/mi_file_util.h"
#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgGLResource/mi_gl_context.h"

#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

#include "MedImgAppCommon/mi_app_cell.h"
#include "MedImgAppCommon//mi_app_thread_model.h"
#include "MedImgAppCommon/mi_app_common_define.h"

#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_review_controller.h"

MED_IMG_BEGIN_NAMESPACE

LoadSeriesCommandHandler::LoadSeriesCommandHandler(std::shared_ptr<ReviewController> controller):_controller(controller)
{

}

LoadSeriesCommandHandler::~LoadSeriesCommandHandler()
{

}

int LoadSeriesCommandHandler::handle_command(const IPCDataHeader& ipcheader , void* buffer)
{
    std::shared_ptr<ReviewController> controller = _controller.lock();
    if(nullptr == controller){
        REVIEW_THROW_EXCEPTION("controller pointer is null!");
    }

    std::shared_ptr<AppThreadModel> thread_model = controller->get_thread_model();
    REVIEW_CHECK_NULL_EXCEPTION(thread_model);

    std::shared_ptr<GLContext> gl_context = thread_model->get_gl_context();

    gl_context->make_current(MAIN_CONTEXT);

    CHECK_GL_ERROR;

    const unsigned int cell_id = ipcheader._msg_info0;
    const unsigned int op_id = ipcheader._msg_info1;

    //1 load series
    const std::string series_path("/home/wr/data/AB_CTA_01/");
    std::vector<std::string> postfix(3);
    postfix[0] = ".dcm";
    postfix[1] = ".DCM";
    postfix[2] = ".Dcm";
    std::vector<std::string> dcm_files;

    FileUtil::get_all_file_recursion(series_path , postfix , dcm_files);

    if(dcm_files.empty()){
        REVIEW_THROW_EXCEPTION("Empty series files!");
    }

    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> img_data;
    DICOMLoader loader;
    IOStatus status = loader.load_series(dcm_files, img_data , data_header);
    if (status != IO_SUCCESS){
        REVIEW_THROW_EXCEPTION("load series failed");
    }

    //2 construct volume infos
    std::shared_ptr<VolumeInfos> volume_infos(new VolumeInfos());
    volume_infos->set_data_header(data_header);
    //SharedWidget::instance()->makeCurrent();
    volume_infos->set_volume(img_data);//load volume texture if has graphic card

    //Create empty mask
    std::shared_ptr<ImageData> mask_data(new ImageData());
    img_data->shallow_copy(mask_data.get());
    mask_data->_channel_num = 1;
    mask_data->_data_type = medical_imaging::UCHAR;
    mask_data->mem_allocate();
    volume_infos->set_mask(mask_data);

    controller->set_volume_infos(volume_infos);

    //3 construct cell
    std::shared_ptr<AppCell> cell(new AppCell);
    std::shared_ptr<MPRScene> mpr_scene(new MPRScene(512,512));
    cell->set_scene(mpr_scene);

    const float PRESET_CT_LUNGS_WW = 1500;
    const float PRESET_CT_LUNGS_WL = -400;

    mpr_scene->set_volume_infos(volume_infos);
    mpr_scene->set_sample_rate(1.0);
    mpr_scene->set_global_window_level(PRESET_CT_LUNGS_WW,PRESET_CT_LUNGS_WL);
    mpr_scene->set_composite_mode(COMPOSITE_AVERAGE);
    mpr_scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
    mpr_scene->set_mask_mode(MASK_NONE);
    mpr_scene->set_interpolation_mode(LINEAR);
    mpr_scene->place_mpr(SAGITTAL);
    controller->add_cell(0 , cell);

    CHECK_GL_ERROR;

    gl_context->make_noncurrent();

    return 0;
}


MED_IMG_END_NAMESPACE
