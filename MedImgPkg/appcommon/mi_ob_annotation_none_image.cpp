#include "mi_ob_annotation_none_image.h"

#include "mi_app_none_image.h"
#include "mi_model_annotation.h"

MED_IMG_BEGIN_NAMESPACE

OBAnnotationNoneImg::OBAnnotationNoneImg() {

}

OBAnnotationNoneImg::~OBAnnotationNoneImg() {

}

void OBAnnotationNoneImg::set_model(std::shared_ptr<ModelAnnotation> model) {
    _model = model;
}

void OBAnnotationNoneImg::set_mpr_none_image(std::vector<AppNoneImage> app_noneimgs) {
    _app_noneimgs = app_noneimgs;
}

void OBAnnotationNoneImg::update(int code_id) {
}

MED_IMG_END_NAMESPACE