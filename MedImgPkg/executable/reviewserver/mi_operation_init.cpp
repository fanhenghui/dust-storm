#include "mi_operation_init.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_client_proxy.h"
#include "util/mi_ipc_server_proxy.h"

#include "arithmetic/mi_run_length_operator.h"

#include "io/mi_dicom_loader.h"
#include "io/mi_image_data.h"
#include "io/mi_image_data_header.h"
#include "io/mi_pacs_communicator.h"
#include "io/mi_nodule_set_parser.h"
#include "io/mi_nodule_set.h"
#include "io/mi_cache_db.h"
#include "io/mi_db.h"
#include "io/mi_configure.h"
#include "io/mi_protobuf.h"

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
#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_none_image.h"
#include "appcommon/mi_model_annotation.h"
#include "appcommon/mi_model_dbs_status.h"
#include "appcommon/mi_model_crosshair.h"
#include "appcommon/mi_model_anonymization.h"
#include "appcommon/mi_ob_annotation_list.h"
#include "appcommon/mi_ob_annotation_segment.h"
#include "appcommon/mi_ob_annotation_statistic.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_common_util.h"

#include "mi_review_logger.h"

#include <time.h>

MED_IMG_BEGIN_NAMESPACE

const float DEFAULT_WW = 1500;
const float DEFAULT_WL = -400;
const std::string LUNG_LUT_PATH = "../config/lut/3d/ct_cta.xml";
const RGBUnit COLOR_TRANSVERSE = RGBUnit(237, 25, 35);
const RGBUnit COLOR_CORONAL = RGBUnit(255, 128, 0);
const RGBUnit COLOR_SAGITTAL = RGBUnit(1, 255, 64);

OpInit::OpInit() {}

OpInit::~OpInit() {}

int OpInit::execute() {
    MI_REVIEW_LOG(MI_TRACE) << "IN init operation.";

    REVIEW_CHECK_NULL_EXCEPTION(_buffer);
    MsgInit msg_init;
    if (0 != protobuf_parse(_buffer, _header.data_len, msg_init)) {
        MI_REVIEW_LOG(MI_ERROR) << "parse init message failed.";
        return -1;
    }

    std::shared_ptr<AppController> controller = get_controller<AppController>();
    REVIEW_CHECK_NULL_EXCEPTION(controller);
    if (0 != init_model(controller)) {
        MI_REVIEW_LOG(MI_FATAL) << "init model failed.";
        return -1;
    }

    //query to get series uid
    //query in remote DB
    std::string db_ip_port,db_user,db_pwd,db_name;
    Configure::instance()->get_db_info(db_ip_port, db_user, db_pwd, db_name);
    DB db;
    if( 0 != db.connect(db_user, db_ip_port, db_pwd, db_name)) {
        MI_REVIEW_LOG(MI_FATAL) << "connect DB failed.";
        return -1;
    }

    std::vector<std::string> series_uids;
    if(0 != db.query_series_uid(msg_init.series_pk(), &series_uids)) {
        MI_REVIEW_LOG(MI_FATAL) << "query series " << msg_init.series_pk() <<  " failed.";
        return -1;
    } else if (series_uids.empty()) {
        MI_REVIEW_LOG(MI_FATAL) << "series " << msg_init.series_pk() <<  " doesn't existed.";
        return -1;
    }

    const std::string series_uid = series_uids[0];

    bool preprocessing_mask = false;
    if (0 != init_data(controller, series_uid, &msg_init, preprocessing_mask)) {
        MI_REVIEW_LOG(MI_FATAL) << "init data failed.";
        return -1;
    }

    if (0 != init_cell(controller, series_uid, &msg_init, preprocessing_mask)) {
        MI_REVIEW_LOG(MI_FATAL) << "init cell failed.";
        return -1;
    }

    msg_init.Clear();
    MI_REVIEW_LOG(MI_TRACE) << "OUT init operation.";
    return 0;
}

int OpInit::init_data(std::shared_ptr<AppController> controller, const std::string& series_uid, MsgInit* msg_init, bool& preprocessing_mask) {
    // reset mask label store
    MaskLabelStore::instance()->reset_labels();
    
    // get series path from img cache db
    MI_REVIEW_LOG(MI_TRACE) << "try to get series from local img cache db. series id: " << series_uid;
    std::string db_ip_port,db_user,db_pwd,db_name,db_path;
    Configure::instance()->get_cache_db_info(db_ip_port, db_user, db_pwd, db_name,db_path);
    CacheDB cache_db;
    if( 0 != cache_db.connect(db_user, db_ip_port, db_pwd, db_name)) {
        MI_REVIEW_LOG(MI_FATAL) << "connect Cache DB failed.";
        return -1;
    }

    std::vector<std::string> instance_file_paths;
    bool dicom_in_cache = false;
    if (0 != cache_db.query_series_instance(series_uid, &instance_file_paths)) {
        MI_REVIEW_LOG(MI_ERROR) << "query cache db failed. try retrieve from remote db.";
    } else {
        if (!instance_file_paths.empty()) {
            MI_REVIEW_LOG(MI_INFO) << "series: " << series_uid << " hit cache.";
            const int err = load_dcm_from_cache_db(controller, instance_file_paths);
            if(-1 == err ) {
                //load series failed
                MI_REVIEW_LOG(MI_FATAL) << "load series :" << series_uid << " failed.";
                return -1;
            } else if (-2 == err) {
                //DB has damaged cache series
                //load from remote to update cache
                dicom_in_cache = false;
            } else {
                MI_REVIEW_LOG(MI_INFO) << "load series from cache db success.";
                dicom_in_cache = true;
            }
        }
    }

    return query_from_remote_db(controller, series_uid, msg_init, dicom_in_cache, preprocessing_mask);
}

int OpInit::load_dcm_from_cache_db(std::shared_ptr<AppController> controller, std::vector<std::string>& instance_file_paths) {
    MI_REVIEW_LOG(MI_TRACE) << "IN load dcm from cache db.";

    //load DICOM
    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> img_data;
    DICOMLoader loader;
    IOStatus status = loader.load_series(instance_file_paths, img_data, data_header);
    if (status != IO_SUCCESS) {
        return -1;
    }

    // create volume infos
    const GPUPlatform gpu_platform = Configure::instance()->get_gpu_platform_type() == CUDA ? CUDA_BASE : GL_BASE;
    std::shared_ptr<VolumeInfos> volume_infos(new VolumeInfos(GPU_BASE, gpu_platform));
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

    MI_REVIEW_LOG(MI_TRACE) << "OUT load dcm from cache db.";
    return 0;
}

static IPCPackage* create_info_msg_package(int op_id, const std::string& series_uid) {
    IPCDataHeader post_header;
    post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    post_header.op_id = op_id;

    MsgString msg_series;
    msg_series.set_context(series_uid);
    char* post_data = nullptr;
    int post_size = 0;
    if (0 != protobuf_serialize(msg_series, post_data, post_size)) {
        return nullptr;
    }
    msg_series.Clear();
    
    post_header.data_len = post_size;
    return (new IPCPackage(post_header,post_data));
}

static IPCPackage* create_query_end_msg_package() {
    IPCDataHeader header;
    header.msg_id = COMMAND_ID_DB_BE_OPERATION;
    header.op_id = OPERATION_ID_DB_BE_REQUEST_END;
    return (new IPCPackage(header));
}

int OpInit::query_from_remote_db(std::shared_ptr<AppController> controller, const std::string& series_uid, MsgInit* msg_init, bool dicom_in_cache, bool& preprocessing_mask) {
    std::string dbs_ip,dbs_port;
    Configure::instance()->get_db_server_host(dbs_ip, dbs_port);
    if (dbs_ip.empty() || dbs_port.empty()) {
        MI_REVIEW_LOG(MI_FATAL) << "DBS host is null. U need check the config file.";
        return -1;
    }

    //use DBS status model to record query situation
    std::shared_ptr<IModel> model = controller->get_model(MODEL_ID_DBS_STATUS);
    std::shared_ptr<ModelDBSStatus> model_dbs_status = std::dynamic_pointer_cast<ModelDBSStatus>(model);
    REVIEW_CHECK_NULL_EXCEPTION(model_dbs_status);
    model_dbs_status->reset();

    IPCClientProxy client_proxy(INET);
    client_proxy.set_server_address(dbs_ip,dbs_port);

    std::vector<IPCPackage*> packages;
    //retrieve DICOM from remote DB
    if (!dicom_in_cache) {
        IPCDataHeader post_header;
        post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
        post_header.op_id = OPERATION_ID_DB_BE_FETCH_DICOM;

        MsgDcmDBRetrieveKey msg;
        msg.set_series_pk(msg_init->series_pk());
        char* post_data = nullptr;
        int post_size = 0;
        if (0 == protobuf_serialize(msg, post_data, post_size)) {
            msg.Clear();
            post_header.data_len = post_size;
            packages.push_back(new IPCPackage(post_header,post_data));    
        } else {
            msg.Clear();
            MI_REVIEW_LOG(MI_ERROR) << "create dcm series retrieve msg failed.";
            return -1;
        }
    }

    //TODO 这里的请求不是通用的，eva_type prep_type 需要根据不同的应用来定义，不过现在一个app的情况下，无所谓
    //retrieve preprocess(init_segment_mask)
    {
        IPCDataHeader post_header;
        post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
        post_header.op_id = OPERATION_ID_DB_BE_FETCH_PREPROCESS_MASK;

        MsgPreprocessRetrieveKey msg;
        msg.set_series_pk(msg_init->series_pk());
        msg.set_prep_type(INIT_SEGMENT_MASK);
        char* post_data = nullptr;
        int post_size = 0;
        if (0 == protobuf_serialize(msg, post_data, post_size)) {
            msg.Clear();
            post_header.data_len = post_size;
            packages.push_back(new IPCPackage(post_header,post_data));    
        } else {
            msg.Clear();
            MI_REVIEW_LOG(MI_ERROR) << "create preprocess retrieve msg failed.";
            return -1;
        }
    }
    //retrieve evaluation(lung nodule)
    {
        IPCDataHeader post_header;
        post_header.msg_id = COMMAND_ID_DB_BE_OPERATION;
        post_header.op_id = OPERATION_ID_DB_BE_FETCH_AI_EVALUATION;

        MsgEvaluationRetrieveKey msg;
        msg.set_series_pk(msg_init->series_pk());
        msg.set_eva_type(LUNG_NODULE);
        char* post_data = nullptr;
        int post_size = 0;
        if (0 == protobuf_serialize(msg, post_data, post_size)) {
            msg.Clear();
            post_header.data_len = post_size;
            packages.push_back(new IPCPackage(post_header,post_data));    
        } else {
            msg.Clear();
            MI_REVIEW_LOG(MI_ERROR) << "create evaluation retrieve msg failed.";
            return -1;
        }
    }
    packages.push_back(create_query_end_msg_package());

    model_dbs_status->query_ai_annotation();
    controller->get_client_proxy_dbs()->sync_send_data(packages);

    //sync: wait until end signal array
    model_dbs_status->await();//TODO 设置超时机制
    model_dbs_status->set_init();
    
    //check errors
    if (!model_dbs_status->success()) {
        std::vector<std::string> dbs_errs = model_dbs_status->get_error_infos();
        std::stringstream ss_err;
        for (size_t i = 0; i < dbs_errs.size(); ++i) {
            ss_err << dbs_errs[i];
            if (i != dbs_errs.size()-1) {
                ss_err << " ; ";
            }
        }
        MI_REVIEW_LOG(MI_FATAL) << "query from db server failed: " << ss_err.str();
        return -1;
    } else {
        MI_REVIEW_LOG(MI_INFO) << "query from db server success";
    }

    //check preprocess mask tag
    preprocessing_mask = model_dbs_status->has_preprocess_mask();
    
    return 0;
}

int OpInit::init_cell(std::shared_ptr<AppController> controller, const std::string& series_uid, MsgInit* msg_init, bool preprocessing_mask) {
    MI_REVIEW_LOG(MI_TRACE) << "IN init operation: cell.";

    std::shared_ptr<VolumeInfos> volume_infos = controller->get_volume_infos();
    REVIEW_CHECK_NULL_EXCEPTION(volume_infos);

    std::shared_ptr<ModelAnnotation> model_annotation = AppCommonUtil::get_model_annotation(controller);
    REVIEW_CHECK_NULL_EXCEPTION(model_annotation);
    std::shared_ptr<ModelCrosshair> model_crosshair = AppCommonUtil::get_model_crosshair(controller);
    REVIEW_CHECK_NULL_EXCEPTION(model_crosshair);
    std::shared_ptr<ModelAnonymization> model_anonymization = AppCommonUtil::get_model_anonymization(controller);
    REVIEW_CHECK_NULL_EXCEPTION(model_anonymization);

    //obs
    std::shared_ptr<OBAnnotationSegment> ob_annotation_segment(new OBAnnotationSegment());
    ob_annotation_segment->set_model(model_annotation);
    ob_annotation_segment->set_volume_infos(volume_infos);
    std::shared_ptr<OBAnnotationStatistic> ob_annotation_statistic(new OBAnnotationStatistic());
    ob_annotation_statistic->set_model(model_annotation);
    ob_annotation_statistic->set_volume_infos(volume_infos);
    std::shared_ptr<OBAnnotationList> ob_annotation_list(new OBAnnotationList());
    ob_annotation_list->set_model(model_annotation);
    ob_annotation_list->set_controller(controller);
    model_annotation->add_observer(ob_annotation_segment);
    model_annotation->add_observer(ob_annotation_statistic);
    model_annotation->add_observer(ob_annotation_list);

    std::vector<ScanSliceType> mpr_scan_types;
    std::vector<std::shared_ptr<MPRScene>> mpr_scenes;
    std::vector<std::shared_ptr<VRScene>> vr_scenes;
    const int expected_fps = Configure::instance()->get_expected_fps();
    const GPUPlatform gpu_platform = Configure::instance()->get_gpu_platform_type() == CUDA ? CUDA_BASE : GL_BASE;
    // create cells
    for (int i = 0; i < msg_init->cells_size(); ++i)
    {
        const MsgCellInfo &cell_info = msg_init->cells(i);
        const int width = cell_info.width();
        const int height = cell_info.height();
        const int direction = cell_info.direction();
        const unsigned int cell_id = static_cast<unsigned int>(cell_info.id());
        const int type_id = cell_info.type();

        std::shared_ptr<AppCell> cell(new AppCell);
        std::shared_ptr<AppNoneImage> none_image(new AppNoneImage());
        cell->set_none_image(none_image);
        if (type_id == 1) { // MPR
            std::shared_ptr<MPRScene> mpr_scene(new MPRScene(width, height, GPU_BASE, gpu_platform));
            mpr_scene->set_mask_label_level(L_64);
            mpr_scenes.push_back(mpr_scene);
            cell->set_scene(mpr_scene);
            mpr_scene->set_test_code(0);
            mpr_scene->set_expected_fps(expected_fps);
            {
                std::stringstream ss;
                ss << "cell_" << cell_id;
                mpr_scene->set_name(ss.str());
            }
            mpr_scene->set_volume_infos(volume_infos);
            mpr_scene->set_sample_step(1.0);
            mpr_scene->set_global_window_level(DEFAULT_WW, DEFAULT_WL);
            mpr_scene->set_composite_mode(COMPOSITE_AVERAGE);
            mpr_scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
            mpr_scene->set_mask_mode(MASK_NONE);
            mpr_scene->set_interpolation_mode(LINEAR);

            switch (direction) {
            case 0:
                mpr_scene->place_mpr(SAGITTAL);
                mpr_scan_types.push_back(SAGITTAL);
                break;
            case 1:
                mpr_scene->place_mpr(CORONAL);
                mpr_scan_types.push_back(CORONAL);
                break;
            default: // 2
                mpr_scene->place_mpr(TRANSVERSE);
                mpr_scan_types.push_back(TRANSVERSE);
                break;
            }

            //none-image
            std::shared_ptr<NoneImgCornerInfos> noneimg_infos(new NoneImgCornerInfos());
            noneimg_infos->set_scene(mpr_scene);
            noneimg_infos->set_model(model_anonymization);
            none_image->add_none_image_item(noneimg_infos);
            
            std::shared_ptr<NoneImgAnnotations> noneimg_annotations(new NoneImgAnnotations());
            noneimg_annotations->set_scene(mpr_scene);
            noneimg_annotations->set_model(model_annotation);
            none_image->add_none_image_item(noneimg_annotations);

            std::shared_ptr<NoneImgCrosshair> noneimg_crosshair(new NoneImgCrosshair());
            noneimg_crosshair->set_scene(mpr_scene);
            noneimg_crosshair->set_model(model_crosshair);
            none_image->add_none_image_item(noneimg_crosshair);

            std::shared_ptr<NoneImgDirection> noneimg_direction(new NoneImgDirection());
            noneimg_direction->set_scene(mpr_scene);
            none_image->add_none_image_item(noneimg_direction);

            std::shared_ptr<NoneImgFrustum> noneimg_frustum(new NoneImgFrustum());
            noneimg_frustum->set_scene(mpr_scene);
            none_image->add_none_image_item(noneimg_frustum);

        } else if (type_id == 2) { // VR
            std::shared_ptr<VRScene> vr_scene(new VRScene(width, height, GPU_BASE, gpu_platform));
            vr_scene->set_mask_label_level(L_64);
            vr_scenes.push_back(vr_scene);
            cell->set_scene(vr_scene);
            vr_scene->set_test_code(0);
            vr_scene->set_expected_fps(expected_fps);
            {
                std::stringstream ss;
                ss << "cell_" << cell_id;
                vr_scene->set_name(ss.str());
            }
            vr_scene->set_navigator_visibility(true);
            vr_scene->set_volume_infos(volume_infos);
            vr_scene->set_sample_step(0.5);
            vr_scene->set_global_window_level(DEFAULT_WW, DEFAULT_WL);
            vr_scene->set_composite_mode(COMPOSITE_DVR);
            vr_scene->set_color_inverse_mode(COLOR_INVERSE_DISABLE);
            vr_scene->set_interpolation_mode(LINEAR);
            vr_scene->set_shading_mode(SHADING_PHONG);
            if(preprocessing_mask) {
                std::vector<unsigned char> vis_labels(1,1);
                vr_scene->set_mask_mode(MASK_MULTI_LABEL);
                vr_scene->set_proxy_geometry(PG_BRICKS);
                vr_scene->set_visible_labels(vis_labels);
            } else {
                vr_scene->set_mask_mode(MASK_NONE);
                vr_scene->set_proxy_geometry(PG_BRICKS);
            }

            // load color opacity
            std::shared_ptr<ColorTransFunc> color;
            std::shared_ptr<OpacityTransFunc> opacity;
            float ww, wl;
            RGBAUnit background;
            Material material;
            if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(LUNG_LUT_PATH, color, opacity, ww, wl, background, material)) {
                MI_REVIEW_LOG(MI_FATAL) << "load lut: " << LUNG_LUT_PATH << " failed.";
                REVIEW_THROW_EXCEPTION("load lut failed");
            }
            vr_scene->set_color_opacity(color, opacity, 0);
            vr_scene->set_ambient_color(1.0f, 1.0f, 1.0f, 0.28f);
            vr_scene->set_material(material, 0);
            vr_scene->set_window_level(ww, wl, 0);
            vr_scene->set_test_code(0);

            if(preprocessing_mask) {
                vr_scene->set_color_opacity(color, opacity, 1);
                vr_scene->set_material(material, 1);
                vr_scene->set_window_level(ww, wl, 1);
            } else {
                vr_scene->set_global_window_level(ww, wl);
            }
        
            //none-image
            std::shared_ptr<NoneImgCornerInfos> noneimg_infos(new NoneImgCornerInfos());
            noneimg_infos->set_scene(vr_scene);
            noneimg_infos->set_model(model_anonymization);
            none_image->add_none_image_item(noneimg_infos);

            std::shared_ptr<NoneImgCrosshair> noneimg_crosshair(new NoneImgCrosshair());
            noneimg_crosshair->set_scene(vr_scene);
            noneimg_crosshair->set_model(model_crosshair);
            none_image->add_none_image_item(noneimg_crosshair);
        } else {
            MI_REVIEW_LOG(MI_FATAL) << "invalid cell type id: " << type_id;
            REVIEW_THROW_EXCEPTION("invalid cell type id!");
        }
        controller->add_cell(cell_id, cell);
    }
    ob_annotation_segment->set_vr_scenes(vr_scenes);
    ob_annotation_segment->set_mpr_scenes(mpr_scenes);

    //notify annotation model (if had load annotation file)
    model_annotation->notify(ModelAnnotation::ADD);

    //crosshair model
    if (mpr_scan_types.size() != 3) {
        MI_REVIEW_LOG(MI_ERROR) << "not 3 MPR in init.";
        REVIEW_THROW_EXCEPTION("not 3 MPR in init.");
        return -1;
    }

    ScanSliceType types[3] = {mpr_scan_types[0], mpr_scan_types[1], mpr_scan_types[2]};
    std::shared_ptr<MPRScene> scenes[3] = {mpr_scenes[0], mpr_scenes[1], mpr_scenes[2]};
    RGBUnit colors[3];
    for (int i = 0; i < 3; ++i) {
        if (types[i] == TRANSVERSE) {
            colors[i] = COLOR_TRANSVERSE;
        } else if (types[i] == CORONAL) {
            colors[i] = COLOR_CORONAL;
        } else if (types[i] == SAGITTAL) {
            colors[i] = COLOR_SAGITTAL;
        }
    }
    model_crosshair->set_mpr_scene(types, scenes, colors);

    MI_REVIEW_LOG(MI_TRACE) << "OUT init operation: cell.";
    return 0;
}

int OpInit::init_model(std::shared_ptr<AppController> controller) {
    MI_REVIEW_LOG(MI_TRACE) << "IN init operation: model.";

    controller->add_model(MODEL_ID_ANNOTATION,
    std::shared_ptr<ModelAnnotation>(new ModelAnnotation()));

    controller->add_model(MODEL_ID_CROSSHAIR,
    std::shared_ptr<ModelCrosshair>(new ModelCrosshair()));

    controller->add_model(MODEL_ID_DBS_STATUS, 
    std::shared_ptr<ModelDBSStatus>(new ModelDBSStatus()));

    MI_REVIEW_LOG(MI_TRACE) << "OUT init operation: model.";
    return 0;
}

MED_IMG_END_NAMESPACE