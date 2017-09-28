#include "mi_annotation_calculator.h"

#include "arithmetic/mi_ortho_camera.h"
#include "arithmetic/mi_arithmetic_utils.h"
#include "io/mi_image_data.h"
#include "mi_camera_calculator.h"
#include "mi_mpr_scene.h"

MED_IMG_BEGIN_NAMESPACE

Ellipsoid AnnotationCalculator::patient_sphere_to_volume_ellipsoid(
    const VOISphere& voi, 
    std::shared_ptr<ImageData> volume_data, 
    std::shared_ptr<CameraCalculator> camera_cal) {

    const Matrix4& mat_p2w = camera_cal->get_patient_to_world_matrix();
    const Matrix4& mat_w2v = camera_cal->get_world_to_volume_matrix();
    Matrix4 mat_p2v = mat_w2v*mat_p2w;

    PatientAxisInfo head_info = camera_cal->get_head_patient_axis_info();
    PatientAxisInfo posterior_info = camera_cal->get_posterior_patient_axis_info();
    PatientAxisInfo left_info = camera_cal->get_left_patient_axis_info();
    double basic_abc[3];
    basic_abc[head_info.volume_coord/2] = volume_data->_spacing[head_info.volume_coord/2];
    basic_abc[posterior_info.volume_coord/2] = volume_data->_spacing[posterior_info.volume_coord/2];
    basic_abc[left_info.volume_coord/2] = volume_data->_spacing[left_info.volume_coord/2];

    Ellipsoid ellipsoid;
    ellipsoid._center = mat_p2v.transform(voi.center);
    double voi_abc[3] = {0,0,0};
    voi_abc[head_info.volume_coord/2] = voi.diameter*0.5/basic_abc[head_info.volume_coord/2] ;
    voi_abc[left_info.volume_coord/2] = voi.diameter*0.5/basic_abc[left_info.volume_coord/2] ;
    voi_abc[posterior_info.volume_coord/2] = voi.diameter*0.5/basic_abc[posterior_info.volume_coord/2] ;
    ellipsoid._a = voi_abc[0];
    ellipsoid._b = voi_abc[1];
    ellipsoid._c = voi_abc[2];

    return ellipsoid;
}

bool AnnotationCalculator::patient_sphere_to_dc_circle(
    const VOISphere& voi, 
    std::shared_ptr<CameraCalculator> camera_cal, 
    std::shared_ptr<MPRScene> scene,
    Circle& circle) {
    
    int width(0), height(0);
    scene->get_display_size(width, height);
    std::shared_ptr<CameraBase> camera = scene->get_camera();
    Point3 look_at = camera->get_look_at();
    Point3 eye = camera->get_eye();
    Vector3 norm = look_at - eye;
    norm.normalize();
    Vector3 up = camera->get_up_direction();

    const Matrix4 mat_vp = camera->get_view_projection_matrix();
    const Matrix4 mat_p2w = camera_cal->get_patient_to_world_matrix();

    std::vector<Point2> circle_center;
    std::vector<float> radiuses;
    std::vector<int> voi_id;
    Point3 sphere_center = voi.center;
    double diameter = voi.diameter;
    sphere_center = mat_p2w.transform(sphere_center);
    double distance = norm.dot_product(look_at - sphere_center);
    if (fabs(distance) < diameter*0.5)
    {
        Point3 pt0 = sphere_center + distance*norm;
        double radius = sqrt(diameter*diameter*0.25 - distance*distance);
        Point3 pt1 = pt0 + radius*up;
        pt0 = mat_vp.transform(pt0);
        pt1 = mat_vp.transform(pt1);
        Point2 pt_dc0 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt0.x , pt0.y) , width , height);
        Point2 pt_dc1 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt1.x , pt1.y) , width , height);
        float radius_float = static_cast<float>( (pt_dc1 - pt_dc0).magnitude() );
        if (radius_float > 0)
        {
            circle._center = pt_dc0;
            circle._radius = radius_float;
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}

bool AnnotationCalculator::dc_circle_update_to_patient_sphere(
    const Circle& circle,
    std::shared_ptr<CameraCalculator> camera_cal, 
    std::shared_ptr<MPRScene> scene, 
    VOISphere& voi) {
    
    int width(0), height(0);
    scene->get_display_size(width, height);
    std::shared_ptr<CameraBase> camera = scene->get_camera();
    Point3 look_at = camera->get_look_at();
    Point3 eye = camera->get_eye();
    Vector3 norm = look_at - eye;
    norm.normalize();
    Vector3 up = camera->get_up_direction();

    const Matrix4 mat_vp = camera->get_view_projection_matrix();
    const Matrix4 mat_vp_inv = mat_vp.get_inverse();
    const Matrix4 mat_p2w = camera_cal->get_patient_to_world_matrix();

    VOISphere pre_voi = voi;
    //1 update ceneter
    //Calculate current circle center world
    Point2 cur_circle_center_ndc = ArithmeticUtils::dc_to_ndc(circle._center, width, height);
    Point3 cur_circle_center = mat_vp_inv.transform( Point3(cur_circle_center_ndc.x, cur_circle_center_ndc.y , 0.0));

    //Calculate previous circle center world
    Point3 sphere_center = mat_p2w.transform(pre_voi.center);
    sphere_center = mat_vp.transform(sphere_center);
    sphere_center.z = 0.0;
    Point3 pre_circle_center = mat_vp_inv.transform(sphere_center);

    Vector3 translate = cur_circle_center - pre_circle_center;
    Point3 new_center = pre_voi.center + translate;
    voi.center = new_center;

    //2 update radius
    const double diameter = pre_voi.diameter;
    const double distance = norm.dot_product(look_at - sphere_center);
    Point3 pt0 = sphere_center + distance*norm;
    const double radius = sqrt(diameter*diameter*0.25 - distance*distance);
    Point3 pt1 = pt0 + radius*up;
    pt0 = mat_vp.transform(pt0);
    pt1 = mat_vp.transform(pt1);
    const Point2 pt_dc0 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt0.x , pt0.y) , width , height);
    const Point2 pt_dc1 = ArithmeticUtils::ndc_to_dc_decimal(Point2(pt1.x , pt1.y) , width , height);
    const double pre_radius = (pt_dc1 - pt_dc0).magnitude();

    //Get current circle radius
    const double cur_radius = circle._radius;
    const double radio = fabs(cur_radius / pre_radius);
    double new_radius = pre_voi.diameter*radio;
    voi.diameter = new_radius;

    return true;
}

MED_IMG_END_NAMESPACE