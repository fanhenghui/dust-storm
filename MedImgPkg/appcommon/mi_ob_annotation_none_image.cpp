#include "mi_ob_annotation_none_image.h"

#include "mi_app_none_image.h"
#include "mi_model_annotation.h"
#include "mi_app_common_logger.h"

MED_IMG_BEGIN_NAMESPACE

OBAnnotationNoneImg::OBAnnotationNoneImg() {

}

OBAnnotationNoneImg::~OBAnnotationNoneImg() {

}

void OBAnnotationNoneImg::set_model(std::shared_ptr<ModelAnnotation> model) {
    _model = model;
}

void OBAnnotationNoneImg::set_mpr_none_image(std::vector<std::shared_ptr<AppNoneImage>> app_noneimgs) {
    _app_noneimgs = app_noneimgs;
}

void OBAnnotationNoneImg::update(int code_id) {
    std::shared_ptr<ModelAnnotation> model = _model.lock();
    APPCOMMON_CHECK_NULL_EXCEPTION(model);

    //const int process_anno_id = model->get_notify_cache();//TODO use model cache to indetify dirty info(id ? processing way ?)
    for (size_t i = 0; i < _app_noneimgs.size(); i++) {
        //just update annitation none image and dont fill msg to FE
        std::shared_ptr<INoneImg> none_img = _app_noneimgs[i]->get_none_image_item(Annotation);
        if (!none_img) {
            MI_APPCOMMON_LOG(MI_WARNING) << "cant find annotation none image.";
            continue;
        }
        std::shared_ptr<NoneImgAnnotations> anno_none_img = std::dynamic_pointer_cast<NoneImgAnnotations>(none_img);
        if (!anno_none_img) {
            MI_APPCOMMON_LOG(MI_ERROR) << "annotation none image type error.";
            continue;
        }
        anno_none_img->check_dirty();
        anno_none_img->update();
    }
}

MED_IMG_END_NAMESPACE