#include "mi_operation_init.h"
#include "mi_message.pb.h"

#include "util/mi_file_util.h"
#include "util/mi_ipc_client_proxy.h"

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

#include "appcommon//mi_app_thread_model.h"
#include "appcommon/mi_app_cell.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_database.h"
#include "appcommon/mi_app_controller.h"
#include "appcommon/mi_app_none_image.h"
#include "appcommon/mi_model_annotation.h"
#include "appcommon/mi_ob_annotation_list.h"
#include "appcommon/mi_ob_annotation_segment.h"
#include "appcommon/mi_ob_annotation_statistic.h"
#include "appcommon/mi_model_crosshair.h"
#include "appcommon/mi_app_common_define.h"
#include "appcommon/mi_app_common_util.h"

#include "mi_review_config.h"
#include "mi_review_logger.h"

#include <time.h>

MED_IMG_BEGIN_NAMESPACE

const float DEFAULT_WW = 1500;
const float DEFAULT_WL = -400;
const std::string LUNG_NODULE_LUT_PATH = "../config/lut/3d/ct_lung_nodule.xml";
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
    if (!msg_init.ParseFromArray(_buffer, _header._data_len)) {
        MI_REVIEW_LOG(MI_ERROR) << "parse init message failed.";
        return -1;
    }

    std::shared_ptr<AppController> controller = _controller.lock();
    REVIEW_CHECK_NULL_EXCEPTION(controller);

    if (0 != init_model_i(controller, &msg_init)) {
        MI_REVIEW_LOG(MI_FATAL) << "init model failed.";
        return -1;
    }

    bool preprocessing_mask = false;
    if (0 != init_data_i(controller, &msg_init, preprocessing_mask)) {
        MI_REVIEW_LOG(MI_FATAL) << "init data failed.";
        return -1;
    }

    if (0 != init_cell_i(controller, &msg_init, preprocessing_mask)) {
        MI_REVIEW_LOG(MI_FATAL) << "init cell failed.";
        return -1;
    }

    msg_init.Clear();
    MI_REVIEW_LOG(MI_TRACE) << "OUT init operation.";
    return 0;
}

int OpInit::init_data_i(std::shared_ptr<AppController> controller, MsgInit* msg_init, bool& preprocessing_mask) {
    MI_REVIEW_LOG(MI_TRACE) << "IN init operation: data.";
    // load data
    // get series path from img cache db
    const std::string series_uid = msg_init->series_uid();
    MI_REVIEW_LOG(MI_TRACE) << "try to get series from local img cache db. series id: " << series_uid;
    AppDB db;
    const std::string db_wpd = ReviewConfig::instance()->get_db_pwd();
    if (0 != db.connect("root", "127.0.0.1:3306", db_wpd.c_str(), "med_img_cache_db")) {
        MI_REVIEW_LOG(MI_ERROR) << "connect to local img cache db failed.";
        return -1;
    }

    std::string data_path;
    if (0 != db.get_series_path(series_uid, data_path)) {
        MI_REVIEW_LOG(MI_ERROR) << "get series: " << series_uid << " 's path failed.";
        return -1;
    }
    MI_REVIEW_LOG(MI_TRACE) << "get series from local img cache db success. data path: " << data_path; 
    const std::string series_path(data_path);

    //get dcm files
    std::vector<std::string> dcm_files;
    std::set<std::string> postfix;
    postfix.insert(".dcm");
    FileUtil::get_all_file_recursion(series_path, postfix, dcm_files);
    if (dcm_files.empty()) {
        MI_REVIEW_LOG(MI_ERROR) << "series path has no DICOM(.dcm) files.";
        return -1;
    }

    //load DICOM
    std::shared_ptr<ImageDataHeader> data_header;
    std::shared_ptr<ImageData> img_data;
    DICOMLoader loader;
    IOStatus status = loader.load_series(dcm_files, img_data, data_header);
    if (status != IO_SUCCESS) {
        MI_REVIEW_LOG(MI_ERROR) << "load series :" << series_uid << " failed.";
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

    // serach preprocessing mask(rle)
    const unsigned int image_buffer_size = mask_data->_dim[0]*mask_data->_dim[1]*mask_data->_dim[2];
    std::set<std::string> mask_postfix;
    mask_postfix.insert(".rle");
    std::vector<std::string> mask_file;
    FileUtil::get_all_file_recursion(series_path, mask_postfix, mask_file);
    preprocessing_mask = false;
    if (!mask_file.empty()) {
        const std::string rle_file = mask_file[0];
        if (0 != RunLengthOperator::decode(rle_file, (unsigned char*)mask_data->get_pixel_pointer(), image_buffer_size)) {
            //TODO do preprocessing
            memset((char*)mask_data->get_pixel_pointer(), 1, image_buffer_size);
            MaskLabelStore::instance()->fill_label(1);
            volume_infos->cache_original_mask();
        } else {
            //TODO check mask label(or just set as 1)
            MaskLabelStore::instance()->fill_label(1);
            volume_infos->cache_original_mask();
            MI_REVIEW_LOG(MI_DEBUG) << "read original rle mask success.";
        }
        preprocessing_mask = true;
    } else {
        //TODO TMP set all mask pixel to 1 (replaced by preprocessing algorithm later)
        unsigned char* mask_raw = (unsigned char*)mask_data->get_pixel_pointer();
        memset(mask_raw, 1 , image_buffer_size);
        MaskLabelStore::instance()->fill_label(1);
        volume_infos->cache_original_mask();
        preprocessing_mask = true;
    }

    controller->set_volume_infos(volume_infos);

    //load annotation file
    std::set<std::string> annotation_postfix;
    annotation_postfix.insert(".csv");
    std::vector<std::string> annotation_file;
    FileUtil::get_all_file_recursion(series_path, annotation_postfix, annotation_file);
    if (!annotation_file.empty()) {
        std::shared_ptr<ModelAnnotation> model_annotation = AppCommonUtil::get_model_annotation(controller);
        REVIEW_CHECK_NULL_EXCEPTION(model_annotation);

        NoduleSetParser parser;
        std::shared_ptr<NoduleSet> nodule_set(new NoduleSet());
        if (IO_SUCCESS != parser.load_as_csv(annotation_file[0], nodule_set) ) {
            MI_REVIEW_LOG(MI_ERROR) << "load annotation file: " << annotation_file[0] << " faild.";
        } else {
            const std::vector<VOISphere>& vois = nodule_set->get_nodule_set();
            std::vector<std::string> ids;
            const float possibility_threshold = ReviewConfig::instance()->get_nodule_possibility_threshold();
            for (size_t i = 0; i < vois.size(); ++i) {
                if (vois[i].para0 < possibility_threshold) {
                    continue;
                }
                std::stringstream ss;
                ss << clock() << '|' << i; 
                const std::string id = ss.str();
                ids.push_back(id); 
                unsigned char new_label = MaskLabelStore::instance()->acquire_label();
                model_annotation->add_annotation(vois[i], id, new_label);
            }
            model_annotation->set_processing_cache(ids);
        }
    }

    MI_REVIEW_LOG(MI_TRACE) << "OUT init operation: data.";
    return 0;
}

int OpInit::init_cell_i(std::shared_ptr<AppController> controller, MsgInit* msg_init, bool preprocessing_mask) {
    MI_REVIEW_LOG(MI_TRACE) << "IN init operation: cell.";

    std::shared_ptr<VolumeInfos> volume_infos = controller->get_volume_infos();
    REVIEW_CHECK_NULL_EXCEPTION(volume_infos);

    std::shared_ptr<ModelAnnotation> model_annotation = AppCommonUtil::get_model_annotation(controller);
    REVIEW_CHECK_NULL_EXCEPTION(model_annotation);
    std::shared_ptr<ModelCrosshair> model_crosshair = AppCommonUtil::get_model_crosshair(controller);
    REVIEW_CHECK_NULL_EXCEPTION(model_crosshair);

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
    const int expected_fps = ReviewConfig::instance()->get_expected_fps();
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
            std::shared_ptr<MPRScene> mpr_scene(new MPRScene(width, height));
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
            mpr_scene->set_sample_rate(1.0);
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
            std::shared_ptr<VRScene> vr_scene(new VRScene(width, height));
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
            vr_scene->set_sample_rate(1.0);
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

int OpInit::init_model_i(std::shared_ptr<AppController> controller, MsgInit*) {
    MI_REVIEW_LOG(MI_TRACE) << "IN init operation: model.";

    std::shared_ptr<ModelAnnotation> model_annotation(new ModelAnnotation());
    controller->add_model(MODEL_ID_ANNOTATION , model_annotation);

    std::shared_ptr<ModelCrosshair> model_crosshair(new ModelCrosshair());
    controller->add_model(MODEL_ID_CROSSHAIR , model_crosshair);

    MI_REVIEW_LOG(MI_TRACE) << "OUT init operation: model.";
    return 0;
}

MED_IMG_END_NAMESPACE