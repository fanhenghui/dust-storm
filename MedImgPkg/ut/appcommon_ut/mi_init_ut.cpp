#include "log/mi_logger.h"

#include "appcommon/mi_message.pb.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_ipc_server_proxy.h"

#include "arithmetic/mi_run_length_operator.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"
#include "io/mi_pacs_communicator.h"
#include "io/mi_worklist_info.h"
#include "io/mi_nodule_set_parser.h"
#include "io/mi_nodule_set.h"

#include "glresource/mi_gl_context.h"
#include "glresource/mi_gl_utils.h"

#include "renderalgo/mi_mpr_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_vr_scene.h"

#include "renderalgo/mi_color_transfer_function.h"
#include "renderalgo/mi_opacity_transfer_function.h"
#include "renderalgo/mi_transfer_function_loader.h"
#include "renderalgo/mi_mask_label_store.h"

#include "appcommon/mi_app_thread_model.h"
#include "appcommon/mi_app_cell.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_cache_db.h"
#include "appcommon/mi_app_db.h"
#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_none_image.h"
#include "appcommon/mi_model_annotation.h"
#include "appcommon/mi_model_dbs_status.h"
#include "appcommon/mi_model_crosshair.h"
#include "appcommon/mi_ob_annotation_list.h"
#include "appcommon/mi_ob_annotation_segment.h"
#include "appcommon/mi_ob_annotation_statistic.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_common_util.h"
#include "appcommon/mi_cmd_handler_recv_dbs_ai_annotation.h"
#include "appcommon/mi_cmd_handler_recv_dbs_dicom_series.h"
#include "appcommon/mi_cmd_handler_recv_dbs_end_signal.h"
#include "appcommon/mi_cmd_handler_recv_dbs_error.h"
#include "appcommon/mi_cmd_handler_recv_dbs_preprocess_mask.h"

#include "appcommon/mi_app_config.h"

using namespace medical_imaging;

#define MI_APPCOMMONUT_LOG(sev) MI_LOG(sev) << "[APPCOMMON UT] "

const float DEFAULT_WW = 1500;
const float DEFAULT_WL = -400;
const std::string LUNG_NODULE_LUT_PATH = "../config/lut/3d/ct_lung_nodule.xml";
const std::string LUNG_LUT_PATH = "../config/lut/3d/ct_cta.xml";
const RGBUnit COLOR_TRANSVERSE = RGBUnit(237, 25, 35);
const RGBUnit COLOR_CORONAL = RGBUnit(255, 128, 0);
const RGBUnit COLOR_SAGITTAL = RGBUnit(1, 255, 64);

static void init_model(std::shared_ptr<AppController> controller) {
    std::shared_ptr<ModelAnnotation> model_annotation(new ModelAnnotation());
    controller->add_model(MODEL_ID_ANNOTATION , model_annotation);

    std::shared_ptr<ModelCrosshair> model_crosshair(new ModelCrosshair());
    controller->add_model(MODEL_ID_CROSSHAIR , model_crosshair);

    std::shared_ptr<ModelDBSStatus> model_dbs_status(new ModelDBSStatus());
    controller->add_model(MODEL_ID_DBS_STATUS , model_dbs_status);

    MI_APPCOMMONUT_LOG(MI_INFO) << "init model.";
}

int load_dcm_from_cache_db(std::shared_ptr<AppController> controller, const std::string& series_uid, const std::string& local_dcm_path) {
    if (local_dcm_path.empty()) {
        MI_APPCOMMONUT_LOG(MI_ERROR) << "series path null in cache db.";
        return -2;
    }

    const std::string series_path = local_dcm_path;
    
    //get dcm files
    std::vector<std::string> dcm_files;
    std::set<std::string> postfix;
    postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(series_path, postfix, dcm_files);
    if (dcm_files.empty()) {
        MI_APPCOMMONUT_LOG(MI_ERROR) << "series path has no DICOM(.dcm) files.";
        return -2;
    }

    //load DICOM
    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> img_data;
    DICOMLoader loader;
    IOStatus status = loader.load_series(dcm_files, img_data, data_header);
    if (status != IO_SUCCESS) {
        MI_APPCOMMONUT_LOG(MI_FATAL) << "load series :" << series_uid << " failed.";
        return -1;
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

    MI_APPCOMMONUT_LOG(MI_TRACE) << "OUT load dcm from cache db.";
    return 0;
}

static IPCPackage* create_info_msg_package(int op_id, const std::string& series_uid) {
    IPCDataHeader post_header;
    char* post_data = nullptr;

    post_header.msg_id = COMMAND_ID_BE_DB_OPERATION;
    post_header.op_id = op_id;

    MsgString msgSeries;
    msgSeries.set_context(series_uid);
    int post_size = msgSeries.ByteSize();
    post_data = new char[post_size];
    if (!msgSeries.SerializeToArray(post_data, post_size)) {
        return nullptr;
    }
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

static IPCPackage* create_query_end_msg_package() {
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_BE_DB_OPERATION;
    header.op_id = OPERATION_ID_DB_QUERY_END;
    return (new IPCPackage(header));
}

int query_from_remote_db(std::shared_ptr<AppController> controller, const std::string& series_uid, bool data_in_cache, bool& preprocessing_mask) {
    std::string dbs_ip,dbs_port;
    AppConfig::instance()->get_db_server_host(dbs_ip, dbs_port);
    if (dbs_ip.empty() || dbs_port.empty()) {
        MI_APPCOMMONUT_LOG(MI_FATAL) << "DBS host is null. U need check the config file.";
        return -1;
    }

    IPCClientProxy client_proxy(INET);
    client_proxy.set_server_address(dbs_ip,dbs_port);

    if (!data_in_cache) {
        client_proxy.register_command_handler(COMMAND_ID_DB_SEND_DICOM_SERIES, 
        std::shared_ptr<CmdHandlerRecvDBSDCMSeries>(new CmdHandlerRecvDBSDCMSeries(controller)));
    }
    
    client_proxy.register_command_handler(COMMAND_ID_DB_SEND_PREPROCESS_MASK, 
    std::shared_ptr<CmdHandlerRecvDBSPreprocessMask>(new CmdHandlerRecvDBSPreprocessMask(controller)));
    client_proxy.register_command_handler(COMMAND_ID_DB_SEND_AI_ANNOTATION, 
    std::shared_ptr<CmdHandlerRecvDBSAIAnno>(new CmdHandlerRecvDBSAIAnno(controller)));
    
    client_proxy.register_command_handler(COMMAND_ID_DB_SEND_END, 
    std::shared_ptr<CmdHandlerRecvDBSEndSignal>(new CmdHandlerRecvDBSEndSignal(controller)));
    client_proxy.register_command_handler(COMMAND_ID_DB_SEND_ERROR, 
    std::shared_ptr<CmdHandlerRecvDBSError>(new CmdHandlerRecvDBSError(controller)));

    std::vector<IPCPackage*> packages;
    if (!data_in_cache) {
        packages.push_back(create_info_msg_package(OPERATION_ID_DB_QUERY_DICOM, series_uid));
    }
    packages.push_back(create_info_msg_package(OPERATION_ID_DB_QUERY_PREPROCESS_MASK, series_uid));
    packages.push_back(create_info_msg_package(OPERATION_ID_DB_QUERY_AI_ANNOTATION, series_uid));
    packages.push_back(create_query_end_msg_package());

    if(0 != client_proxy.sync_post(packages) ) {
        MI_APPCOMMONUT_LOG(MI_FATAL) << "query from db server failed with exception.";
        return -1;
    }

    //check errors
    std::shared_ptr<IModel> model = controller->get_model(MODEL_ID_DBS_STATUS);
    std::shared_ptr<ModelDBSStatus> model_dbs_status = std::dynamic_pointer_cast<ModelDBSStatus>(model);
    std::vector<std::string> dbs_err = model_dbs_status->get_error_infos();
    if (dbs_err.empty()) {
        MI_APPCOMMONUT_LOG(MI_FATAL) << "query from db server failed.";
        return -1;
    }

    return 0;
}

int init_data(std::shared_ptr<AppController> controller, const std::string& series_uid, bool& preprocessing_mask) {
    // load data
    // get series path from img cache db
    std::string db_ip_port,db_user,db_pwd,db_name,db_path;
    AppConfig::instance()->get_cache_db_info(db_ip_port, db_user, db_pwd, db_name,db_path);
    CacheDB cache_db;
    if( 0 != cache_db.connect(db_user, db_ip_port, db_pwd, db_name)) {
        MI_APPCOMMONUT_LOG(MI_FATAL) << "connect Cache DB failed.";
        return -1;
    }

    CacheDB::ImgItem item;
    bool data_in_cache = false;
    if (0 == cache_db.get_item(series_uid, item)) {
        MI_APPCOMMONUT_LOG(MI_INFO) << "hit dcm in cache db.";
        const int err = load_dcm_from_cache_db(controller, series_uid, item.path);
        if(-1 == err ) {
            //load series failed
            return -1;
        } else if (-2 == err) {
            //DB has damaged cache series
            //load from remote to update cache
            data_in_cache = false;
        } else {
            MI_APPCOMMONUT_LOG(MI_INFO) << "load series from cache db success.";
            data_in_cache = true;
        }
    }

    return query_from_remote_db(controller, series_uid, data_in_cache, preprocessing_mask);
}

int init_ut(int argc, char* argv[]) {
    const std::string log_config_file = AppConfig::instance()->get_log_config_file();
    Logger::instance()->bind_config_file(log_config_file);
    Logger::instance()->set_file_name_format("logs/mi-db-ut-%Y-%m-%d_%H-%M-%S.%N.log");
    Logger::instance()->set_file_direction("");
    Logger::instance()->initialize();

    //testing
    // {
    //     std::string rle_file("/home/wangrui22/data/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.rle");
    //     unsigned int volume_size = 512*512*321;
    //     char* buffer = nullptr;
    //     unsigned int size = 0;
    //     FileUtil::read_raw_ext(rle_file,buffer,size);
    //     unsigned char* mask = new unsigned char[volume_size];
    //     RunLengthOperator::decode((unsigned int*)buffer,size/sizeof(unsigned int), mask, volume_size);
    //     return 0;
    // }

    if (argc != 2) {
        MI_APPCOMMONUT_LOG(MI_FATAL) <<"Lack series ID.";
        return -1;
    }

    const std::string series_id = argv[1];
    std::shared_ptr<AppController> controller(new AppController());

    init_model(controller);

    bool preprocessing_mask = false;
    if (0 != init_data(controller, series_id, preprocessing_mask)) {
        MI_APPCOMMONUT_LOG(MI_FATAL) << "init data failed.";
        return -1;
    }

    MI_APPCOMMONUT_LOG(MI_INFO) << "UT success.";
    return 0;
}