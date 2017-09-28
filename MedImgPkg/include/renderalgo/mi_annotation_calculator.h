#ifndef MEDIMG_RENDERALGO_MI_ANNOTATION_CALCULATOR_H
#define MEDIMG_RENDERALGO_MI_ANNOTATION_CALCULATOR_H

#include "renderalgo/mi_render_algo_export.h"
#include "arithmetic/mi_aabb.h"
#include "arithmetic/mi_ellipsoid.h"
#include "arithmetic/mi_circle.h"

#include "io/mi_voi.h"

MED_IMG_BEGIN_NAMESPACE

class ImageData;
class CameraCalculator;
class MPRScene;
class RenderAlgo_Export AnnotationCalculator {
public:
    static Ellipsoid patient_sphere_to_volume_ellipsoid(
        const VOISphere& voi, 
        std::shared_ptr<ImageData> img,
         std::shared_ptr<CameraCalculator> cameracal);

    static bool patient_sphere_to_dc_circle(
        const VOISphere& voi, 
        std::shared_ptr<CameraCalculator> cameracal,
        std::shared_ptr<MPRScene> scene,
        Circle& circle);

    static bool dc_circle_update_to_patient_sphere(
        const Circle& circle,
        std::shared_ptr<CameraCalculator> cameracal,
        std::shared_ptr<MPRScene> scene,
        VOISphere& voi);
};

MED_IMG_END_NAMESPACE

#endif