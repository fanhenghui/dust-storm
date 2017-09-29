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
#include "appcommon/mi_ob_annotation_none_image.h"
#include "appcommon/mi_ob_annotation_segment.h"
#include "appcommon/mi_ob_annotation_statistic.h"
#include "appcommon/mi_app_common_define.h"

#include "mi_review_config.h"
#include "mi_review_logger.h"

MED_IMG_BEGIN_NAMESPACE

OpInit::OpInit() {}

OpInit::~OpInit() {}

int OpInit::execute() {
    MI_REVIEW_LOG(MI_TRACE) << "IN init operation.";

    std::shared_ptr<AppController> controller = _controller.lock();
    REVIEW_CHECK_NULL_EXCEPTION(controller);

    REVIEW_CHECK_NULL_EXCEPTION(_buffer);
    MsgInit msg_init;

    if (!msg_init.ParseFromArray(_buffer, _header._data_len)) {
        MI_REVIEW_LOG(MI_ERROR) << "parse init message failed.";
        return -1;
    }

    // get series path from img cache db
    const std::string series_uid = msg_init.series_uid();
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


    //model&observer
    std::shared_ptr<IModel> anno_model_ = controller->get_model(MODEL_ID_ANNOTATION);
    std::shared_ptr<ModelAnnotation> anno_model = std::dynamic_pointer_cast<ModelAnnotation>(anno_model_);
    REVIEW_CHECK_NULL_EXCEPTION(anno_model);
    std::shared_ptr<OBAnnotationSegment> ob_annotation_segment(new OBAnnotationSegment());
    ob_annotation_segment->set_model(anno_model);
    ob_annotation_segment->set_volume_infos(volume_infos);
    // std::shared_ptr<OBAnnotationStatistic> ob_annotation_statistic(new OBAnnotationStatistic());
    // ob_annotation_statistic->set_model(anno_model);
    std::shared_ptr<OBAnnotationNoneImg> ob_annotation_noneimg(new OBAnnotationNoneImg());
    ob_annotation_noneimg->set_model(anno_model);

    anno_model->add_observer(ob_annotation_segment);
    //anno_model->add_observer(ob_annotation_statistic);
    anno_model->add_observer(ob_annotation_noneimg);

    std::vector<std::shared_ptr<MPRScene>> mpr_scenes;
    std::vector<std::shared_ptr<VRScene>> vr_scenes;
    std::vector<std::shared_ptr<AppNoneImage>> mpr_none_images;

    //TODO hard coding for demo
    if (series_uid == "1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405") {
        //load mask form disk
        std::shared_ptr<ImageData> mask_data(new ImageData());
        img_data->shallow_copy(mask_data.get());
        mask_data->_channel_num = 1;
        mask_data->_data_type = medical_imaging::UCHAR;
        mask_data->mem_allocate();
        char *mask_raw = (char *)mask_data->get_pixel_pointer();
        const std::string root = ReviewConfig::instance()->get_test_data_root() + "/demo/lung";
        std::ifstream in(root + "/mask.raw", std::ios::in);
        const unsigned int data_len = img_data->_dim[0] * img_data->_dim[1] * img_data->_dim[2];
        if (in.is_open()) {
            in.read(mask_raw, data_len);
            in.close();
        }
        else {
            memset(mask_raw, 1, data_len);
        }
        std::set<unsigned char> target_label_set;
        RunLengthOperator run_length_op;
        std::ifstream in2(root + "/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.rle", std::ios::binary | std::ios::in);
        if (in2.is_open()) {
            in2.seekg(0, in2.end);
            const int code_len = in2.tellg();
            in2.seekg(0, in2.beg);
            unsigned int *code_buffer = new unsigned int[code_len];
            in2.read((char *)code_buffer, code_len);
            in2.close();
            unsigned char *mask_target = new unsigned char[data_len];

            if (0 == run_length_op.decode(code_buffer, code_len / sizeof(unsigned int), mask_target, data_len)) {
                //FileUtil::write_raw(root+"./nodule.raw" , mask_target , data_len);
                printf("load target mask done.\n");
                for (unsigned int i = 0; i < data_len; ++i) {
                    if (mask_target[i] != 0) {
                        mask_raw[i] = mask_target[i] + 1;
                        target_label_set.insert(mask_target[i] + 1);
                    }
                }
            }
            delete[] mask_target;

            //fill mask label store
            MaskLabelStore::instance()->fill_label(1);
            for (auto it = target_label_set.begin(); it != target_label_set.end(); ++it) {
                MaskLabelStore::instance()->fill_label(*it);
            }
        }

        volume_infos->set_mask(mask_data);
        controller->set_volume_infos(volume_infos);

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
            std::shared_ptr<AppNoneImage> none_image(new AppNoneImage());
            cell->set_none_image(none_image);
            if (type_id == 1) { // MPR
                std::shared_ptr<MPRScene> mpr_scene(new MPRScene(width, height));
                mpr_scenes.push_back(mpr_scene);
                cell->set_scene(mpr_scene);
                mpr_scene->set_test_code(0); {
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
                mpr_scene->set_mask_overlay_mode(MASK_OVERLAY_ENABLE);

                std::map<unsigned char, RGBAUnit> mask_overlay_color;
                std::vector<unsigned char> vis_labels;
                std::stringstream sslabel;
                sslabel << "mpr overlay label: ";
                for (auto it = target_label_set.begin(); it != target_label_set.end(); ++it) {
                    sslabel <<  int(*it) << " ";
                    vis_labels.push_back(*it);
                    mask_overlay_color[*it] = RGBAUnit(255, 0, 0);
                }
                MI_REVIEW_LOG(MI_DEBUG) << sslabel.str();
                mpr_scene->set_visible_labels(vis_labels);
                mpr_scene->set_mask_overlay_color(mask_overlay_color);

                switch (direction) {
                case 0:
                    mpr_scene->place_mpr(SAGITTAL);
                    break;
                case 1:
                    mpr_scene->place_mpr(CORONAL);
                    break;
                default: // 2
                    mpr_scene->place_mpr(TRANSVERSE);
                    break;
                }

                //none-image
                std::shared_ptr<NoneImgCornerInfos> noneimg_infos(new NoneImgCornerInfos());
                noneimg_infos->set_scene(mpr_scene);
                none_image->add_none_image_item(noneimg_infos);
                std::shared_ptr<NoneImgAnnotations> noneimg_annotations(new NoneImgAnnotations());
                noneimg_annotations->set_scene(mpr_scene);
                std::shared_ptr<IModel> model_ =  controller->get_model(MODEL_ID_ANNOTATION);
                std::shared_ptr<ModelAnnotation> model = std::dynamic_pointer_cast<ModelAnnotation>(model_);
                noneimg_annotations->set_model(model);
                none_image->add_none_image_item(noneimg_annotations);
                mpr_none_images.push_back(none_image);

            } else if (type_id == 2) {  // VR
                std::shared_ptr<VRScene> vr_scene(new VRScene(width, height));
                vr_scenes.push_back(vr_scene);
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
                vr_scene->set_mask_mode(MASK_MULTI_LABEL);
                vr_scene->set_interpolation_mode(LINEAR);
                vr_scene->set_shading_mode(SHADING_PHONG);
                vr_scene->set_proxy_geometry(PG_BRICKS);

                // load color opacity
                std::string color_opacity_xml =
                    "../config/lut/3d/ct_lung_glass.xml";
                std::shared_ptr<ColorTransFunc> color;
                std::shared_ptr<OpacityTransFunc> opacity;
                float ww, wl;
                RGBAUnit background;
                Material material;

                if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(
                                      color_opacity_xml, color, opacity, ww, wl,
                                      background, material)) {
                    MI_REVIEW_LOG(MI_FATAL) << "load lut: " << color_opacity_xml << " failed.";
                    REVIEW_THROW_EXCEPTION("load lut failed.");
                }

                vr_scene->set_color_opacity(color, opacity, 0);
                vr_scene->set_ambient_color(1.0f, 1.0f, 1.0f, 0.28f);
                vr_scene->set_material(material, 0);
                vr_scene->set_window_level(ww, wl, 0);
                vr_scene->set_test_code(0);

                vr_scene->set_color_opacity(color, opacity, 1);
                vr_scene->set_material(material, 1);
                vr_scene->set_window_level(ww, wl, 1);

                //load nodule lut
                color_opacity_xml = "../config/lut/3d/ct_lung_nodule.xml";
                if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(color_opacity_xml, color, opacity, ww, wl, background, material)) {
                    MI_REVIEW_LOG(MI_FATAL) << "load lut: " << color_opacity_xml << " failed.";
                    //REVIEW_THROW_EXCEPTION("load lut failed.");
                }
                std::vector<unsigned char> vis_labels;
                vis_labels.push_back(1);
                for (auto it = target_label_set.begin(); it != target_label_set.end(); ++it) {
                    vis_labels.push_back(*it);
                    vr_scene->set_color_opacity(color, opacity, *it);
                    vr_scene->set_material(material, *it);
                    vr_scene->set_window_level(ww, wl, *it);
                }
                vr_scene->set_visible_labels(vis_labels);

                //none-image
                std::shared_ptr<NoneImgCornerInfos> noneimg_infos(new NoneImgCornerInfos());
                noneimg_infos->set_scene(vr_scene);
                none_image->add_none_image_item(noneimg_infos);
            } else {
                MI_REVIEW_LOG(MI_FATAL) << "invalid cell type id: " << type_id;
                REVIEW_THROW_EXCEPTION("invalid cell type id!");
            }
            controller->add_cell(cell_id, cell);

            ob_annotation_noneimg->set_mpr_none_image(mpr_none_images);
            ob_annotation_segment->set_vr_scenes(vr_scenes);
            ob_annotation_segment->set_mpr_scenes(mpr_scenes);
        }

        return 0;
    }

    // create empty mask
    std::shared_ptr<ImageData> mask_data(new ImageData());
    img_data->shallow_copy(mask_data.get());
    mask_data->_channel_num = 1;
    mask_data->_data_type = medical_imaging::UCHAR;
    mask_data->mem_allocate();
    volume_infos->set_mask(mask_data);
    controller->set_volume_infos(volume_infos);

    // serach mask(rle)
    const unsigned int image_buffer_size = mask_data->_dim[0]*mask_data->_dim[1]*mask_data->_dim[2];
    std::set<std::string> mask_postfix;
    mask_postfix.insert(".rle");
    std::vector<std::string> mask_file;
    FileUtil::get_all_file_recursion(series_path, mask_postfix, mask_file);
    if (mask_file.empty()) {
        memset((char*)mask_data->get_pixel_pointer(), 1, image_buffer_size);
        MaskLabelStore::instance()->fill_label(1);
    } else {
        const std::string rle_file = mask_file[0];
        if (0 != RunLengthOperator::decode(rle_file, (unsigned char*)mask_data->get_pixel_pointer(), image_buffer_size)) {
            memset((char*)mask_data->get_pixel_pointer(), 1, image_buffer_size);
            MaskLabelStore::instance()->fill_label(1);
        } else {
            //TODO check mask label(or just set as 1)
            MaskLabelStore::instance()->fill_label(1);
            MI_REVIEW_LOG(MI_DEBUG) << "read rle mask success.";
            //FileUtil::write_raw("/home/wangrui22/data/rleunzip.raw" , mask_data->get_pixel_pointer() , image_buffer_size);
        }
     }
     std::vector<unsigned char> vis_labels;
     vis_labels.push_back(1);
    
    // create cells
    for (int i = 0; i < msg_init.cells_size(); ++i)
    {
        const MsgCellInfo &cell_info = msg_init.cells(i);
        const int width = cell_info.width();
        const int height = cell_info.height();
        const int direction = cell_info.direction();
        const int cell_id = cell_info.id();
        const int type_id = cell_info.type();

        const float DEFAULT_WW = 1500;
        const float DEFAULT_WL = -400;

        std::shared_ptr<AppCell> cell(new AppCell);
        std::shared_ptr<AppNoneImage> none_image(new AppNoneImage());
        cell->set_none_image(none_image);
        if (type_id == 1) { // MPR
            std::shared_ptr<MPRScene> mpr_scene(new MPRScene(width, height));
            mpr_scenes.push_back(mpr_scene);
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
            mpr_scene->set_visible_labels(vis_labels);

            switch (direction) {
            case 0:
                mpr_scene->place_mpr(SAGITTAL);
                break;
            case 1:
                mpr_scene->place_mpr(CORONAL);
                break;
            default: // 2
                mpr_scene->place_mpr(TRANSVERSE);
                break;
            }

            //none-image
            std::shared_ptr<NoneImgCornerInfos> noneimg_infos(new NoneImgCornerInfos());
            noneimg_infos->set_scene(mpr_scene);
            none_image->add_none_image_item(noneimg_infos);
            std::shared_ptr<NoneImgAnnotations> noneimg_annotations(new NoneImgAnnotations());
            noneimg_annotations->set_scene(mpr_scene);
            std::shared_ptr<IModel> model_ =  controller->get_model(MODEL_ID_ANNOTATION);
            std::shared_ptr<ModelAnnotation> model = std::dynamic_pointer_cast<ModelAnnotation>(model_);
            noneimg_annotations->set_model(model);
            none_image->add_none_image_item(noneimg_annotations);
            mpr_none_images.push_back(none_image);
        } else if (type_id == 2) { // VR
            std::shared_ptr<VRScene> vr_scene(new VRScene(width, height));
            vr_scenes.push_back(vr_scene);
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
            vr_scene->set_mask_mode(MASK_MULTI_LABEL);
            vr_scene->set_interpolation_mode(LINEAR);
            vr_scene->set_shading_mode(SHADING_PHONG);
            vr_scene->set_proxy_geometry(PG_BRICKS);
            vr_scene->set_visible_labels(vis_labels);

            // load color opacity
            const std::string color_opacity_xml = "../config/lut/3d/ct_lung_glass.xml";
            std::shared_ptr<ColorTransFunc> color;
            std::shared_ptr<OpacityTransFunc> opacity;
            float ww, wl;
            RGBAUnit background;
            Material material;

            if (IO_SUCCESS != TransferFuncLoader::load_color_opacity(
                                  color_opacity_xml, color, opacity, ww, wl,
                                  background, material)) {
                MI_REVIEW_LOG(MI_FATAL) << "load lut: " << color_opacity_xml << " failed.";
                REVIEW_THROW_EXCEPTION("load lut failed");
            }
            vr_scene->set_color_opacity(color, opacity, 0);
            vr_scene->set_ambient_color(1.0f, 1.0f, 1.0f, 0.28f);
            vr_scene->set_material(material, 0);
            vr_scene->set_window_level(ww, wl, 0);
            vr_scene->set_test_code(0);

            vr_scene->set_color_opacity(color, opacity, 1);
            vr_scene->set_material(material, 1);
            vr_scene->set_window_level(ww, wl, 1);

            //none-image
            std::shared_ptr<NoneImgCornerInfos> noneimg_infos(new NoneImgCornerInfos());
            noneimg_infos->set_scene(vr_scene);
            none_image->add_none_image_item(noneimg_infos);
        } else {
            MI_REVIEW_LOG(MI_FATAL) << "invalid cell type id: " << type_id;
            REVIEW_THROW_EXCEPTION("invalid cell type id!");
        }
        controller->add_cell(cell_id, cell);

        ob_annotation_noneimg->set_mpr_none_image(mpr_none_images);
        ob_annotation_segment->set_vr_scenes(vr_scenes);
        ob_annotation_segment->set_mpr_scenes(mpr_scenes);

    }

    MI_REVIEW_LOG(MI_TRACE) << "OUT init operation.";
    return 0;
}

MED_IMG_END_NAMESPACE