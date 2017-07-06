#include "mi_load_series_command_handler.h"

#include "MedImgUtil/mi_file_util.h"
#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgGLResource/mi_gl_context.h"

#include "MedImgRenderAlgorithm/mi_volume_infos.h"

#include "mi_review_controller.h"

MED_IMG_BEGIN_NAMESPACE

LoadSeriesCommandHandler::LoadSeriesCommandHandler(std::shared_ptr<AppController> controller):_controller(controller)
{

}

LoadSeriesCommandHandler::~LoadSeriesCommandHandler()
{

}

int LoadSeriesCommandHandler::handle_command(const IPCDataHeader& ipcheader , void* buffer)
{
    std::shared_ptr<AppController> controller = _controller.lock();
    if(nullptr == controller){
        REVIEW_THROW_EXCEPTION("controller pointer is null!");
    }
    std::shared_ptr<GLContext> gl_context = controller->get_gl_context();

    gl_context->make_current(MAIN_CONTEXT);

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

    IOStatus status = loader.load_series(file_names_std, img_data , data_header);
    if (status != IO_SUCCESS){
        REVIEW_THROW_EXCEPTION("load series failed");
    }

    //2 construct volume infos
    if (_volume_infos)//Delete last one
    {
        _volume_infos->finialize();
    }
    _volume_infos.reset(new VolumeInfos());
    _volume_infos->set_data_header(data_header);
    //SharedWidget::instance()->makeCurrent();
    _volume_infos->set_volume(img_data);//load volume texture if has graphic card

    //Create empty mask
    std::shared_ptr<ImageData> mask_data(new ImageData());
    img_data->shallow_copy(mask_data.get());
    mask_data->_channel_num = 1;
    mask_data->_data_type = medical_imaging::UCHAR;
    mask_data->mem_allocate();
    _volume_infos->set_mask(mask_data);


    //2 initialize data

    gl_context->make_noncurrent();

    return 0;
}


MED_IMG_END_NAMESPACE
