#include "mi_camera_calculator.h"
#include "arithmetic/mi_arithmetic_utils.h"
#include "io/mi_image_data.h"
#include "mi_render_algo_logger.h"

MED_IMG_BEGIN_NAMESPACE

CameraCalculator::CameraCalculator(std::shared_ptr<ImageData> image_data)
    : _volume_data(image_data), _is_volume_orthogonal(true) {
    calculate();
}

const Matrix4& CameraCalculator::get_volume_to_physical_matrix() const {

    return _mat_volume_to_physical;
}

const Matrix4& CameraCalculator::get_physical_to_world_matrix() const {
    return _mat_physical_to_world;
}

const Matrix4& CameraCalculator::get_world_to_patient_matrix() const {
    return _mat_world_to_patient;
}

const Matrix4& CameraCalculator::get_volume_to_world_matrix() const {
    return _mat_volume_to_wolrd;
}

const Matrix4& CameraCalculator::get_world_to_volume_matrix() const {
    return _mat_world_to_volume;
}

void CameraCalculator::init_mpr_placement(std::shared_ptr<OrthoCamera> camera,
        ScanSliceType type,
        const Point3& ptCenterPoint) const {
    *camera = _othogonal_mpr_camera[static_cast<int>(type)];
    camera->set_look_at(ptCenterPoint);
}

void CameraCalculator::init_mpr_placement(std::shared_ptr<OrthoCamera> camera,
        ScanSliceType type) const {
    *camera = _othogonal_mpr_camera[static_cast<int>(type)];
}

void CameraCalculator::init_vr_placement(
    std::shared_ptr<OrthoCamera> camera) const {
    *camera = _vr_placement_camera;
}

void CameraCalculator::calculate() {
    // Check orthogonal
    // 对于CT机架倾斜的数据来说，其扫描的X和Y方向并不正交，渲染需要特殊处理
    check_volume_orthogonal();

    //计算体数据的每个轴和标准病人坐标系下的轴的关系
    calculate_patient_axis_info();

    // Calculate volume to physical/world/patient
    calculate_matrix();

    // Calculate VR replacement
    calculate_vr_placement();

    // Calculate orthogonal MPR replacement
    calculate_default_mpr_center_world();
    caluculate_orthogonal_mpr_placement();
}

void CameraCalculator::calculate_matrix() {
    // 1 Calculate volume to physical
    unsigned int* dim = _volume_data->_dim;
    double* spacing = _volume_data->_spacing;

    _mat_volume_to_physical.set_idintity();
    _mat_volume_to_physical.prepend(
        make_translate(-Vector3(dim[0] * 0.5, dim[1] * 0.5, dim[2] * 0.5)));
    _mat_volume_to_physical.prepend(
        make_scale(Vector3(spacing[0], spacing[1], spacing[2])));

    // 2 Calculate physical to world
    // MatA2B * PA = PB; -----> MatA2B = PB*Inv(PA);
    Vector3 standard_coord_axis[6] = {
        Vector3(1.0, 0.0, 0.0), Vector3(-1.0, 0.0, 0.0),
        Vector3(0.0, 1.0, 0.0), Vector3(0.0, -1.0, 0.0),
        Vector3(0.0, 0.0, 1.0), Vector3(0.0, 0.0, -1.0)
    };

    const Vector3 physical_head =
        standard_coord_axis[(int)(_headInfo.volume_coord)];
    const Vector3 physical_posterior =
        standard_coord_axis[(int)(_posteriorInfo.volume_coord)];
    const Vector3 physical_left =
        standard_coord_axis[(int)(_leftInfo.volume_coord)];

    const Matrix4 mat_physical(physical_head.x, physical_head.y, physical_head.z,
                               0.0, physical_posterior.x, physical_posterior.y,
                               physical_posterior.z, 0.0, physical_left.x,
                               physical_left.y, physical_left.z, 0.0, 0.0, 0.0,
                               0.0, 1.0);

    const Vector3 patient_head = _headInfo.patient_orientation;
    const Vector3 patient_posterior = _posteriorInfo.patient_orientation;
    const Vector3 patient_left = _leftInfo.patient_orientation;

    const Matrix4 mat_patient(
        patient_head.x, patient_head.y, patient_head.z, 0.0, patient_posterior.x,
        patient_posterior.y, patient_posterior.z, 0.0, patient_left.x,
        patient_left.y, patient_left.z, 0.0, 0.0, 0.0, 0.0, 1.0);

    if (!mat_physical.has_inverse()) {
        _mat_physical_to_world.set_idintity();
    } else {
        _mat_physical_to_world = mat_patient * mat_physical.get_inverse();
    }

    // 3 Calculate volume to world
    _mat_volume_to_wolrd = _mat_physical_to_world * _mat_volume_to_physical;
    _mat_world_to_volume = _mat_volume_to_wolrd.get_inverse();

    // 4 Calculate world to patient
    const Point3& img_position = _volume_data->_image_position;
    const Point3& img_world =
        _mat_volume_to_wolrd.transform(Point3::S_ZERO_POINT);
    _mat_world_to_patient = make_translate(img_position - img_world);
    _mat_patient_to_world = _mat_world_to_patient.get_inverse();
}

void CameraCalculator::calculate_patient_axis_info() {
    const Vector3& x_coord_patient = _volume_data->_image_orientation[0];
    const Vector3& y_coord_patient = _volume_data->_image_orientation[1];
    const Vector3& z_coord_patient = _volume_data->_image_orientation[2];

    /// rotate the volume to get consistent with patient coordinate
    const Vector3 standard_haed_axis(0.0, 0.0, 1.0);
    const Vector3 standard_left_axis(1.0, 0.0, 0.0);
    const Vector3 standard_posterior_axis(0.0, 1.0, 0.0);

    const double dot_head_x = x_coord_patient.dot_product(standard_haed_axis);
    const double dot_head_y = y_coord_patient.dot_product(standard_haed_axis);
    const double dot_head_z = z_coord_patient.dot_product(standard_haed_axis);

    /// Priority is TRA>COR>SAG.
    if ((!(abs(dot_head_z) - abs(dot_head_x) < 0)) &&
            (!(abs(dot_head_z) - abs(dot_head_y) < 0))) {
        _headInfo.patient_orientation =
            dot_head_z > 0 ? z_coord_patient : -z_coord_patient;
        _headInfo.volume_coord = dot_head_z > 0 ? POSZ : NEGZ;

        const double dot_left_x = x_coord_patient.dot_product(standard_left_axis);
        const double dot_left_y = y_coord_patient.dot_product(standard_left_axis);

        if (!(abs(dot_left_x) < abs(dot_left_y))) {
            _leftInfo.patient_orientation =
                dot_left_x > 0 ? x_coord_patient : -x_coord_patient;
            _leftInfo.volume_coord = dot_left_x > 0 ? POSX : NEGX;

            const double dot_posterior_y =
                y_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation =
                dot_posterior_y > 0 ? y_coord_patient : -y_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_y > 0 ? POSY : NEGY;
        } else {
            _leftInfo.patient_orientation =
                dot_left_y > 0 ? y_coord_patient : -y_coord_patient;
            _leftInfo.volume_coord = dot_left_y > 0 ? POSY : NEGY;

            const double dot_posterior_x =
                x_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation =
                dot_posterior_x > 0 ? x_coord_patient : -x_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_x > 0 ? POSX : NEGX;
        }

    } else if ((!(abs(dot_head_y) - abs(dot_head_x) < 0)) &&
               (!(abs(dot_head_y) - abs(dot_head_z) < 0))) {
        _headInfo.patient_orientation =
            dot_head_y > 0 ? y_coord_patient : -y_coord_patient;
        _headInfo.volume_coord = dot_head_y > 0 ? POSY : NEGY;

        const double dot_left_x = x_coord_patient.dot_product(standard_left_axis);
        const double dot_left_z = z_coord_patient.dot_product(standard_left_axis);

        if (!(abs(dot_left_x) < abs(dot_left_z))) {
            _leftInfo.patient_orientation =
                dot_left_x > 0 ? x_coord_patient : -x_coord_patient;
            _leftInfo.volume_coord = dot_left_x > 0 ? POSX : NEGX;

            const double dot_posterior_z =
                z_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation =
                dot_posterior_z > 0 ? z_coord_patient : -z_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_z > 0 ? POSZ : NEGZ;
        } else {
            _leftInfo.patient_orientation =
                dot_left_z > 0 ? z_coord_patient : -z_coord_patient;
            _leftInfo.volume_coord = dot_left_z > 0 ? POSZ : NEGZ;

            const double dot_posterior_x =
                x_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation =
                dot_posterior_x > 0 ? x_coord_patient : -x_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_x > 0 ? POSX : NEGX;
        }

    } else {
        _headInfo.patient_orientation =
            dot_head_x > 0 ? x_coord_patient : -x_coord_patient;
        _headInfo.volume_coord = dot_head_x > 0 ? POSX : NEGX;

        const double dot_left_y = y_coord_patient.dot_product(standard_left_axis);
        const double dot_left_z = z_coord_patient.dot_product(standard_left_axis);

        if (!(abs(dot_left_y) < abs(dot_left_z))) {
            _leftInfo.patient_orientation =
                dot_left_y > 0 ? y_coord_patient : -y_coord_patient;
            _leftInfo.volume_coord = dot_left_y > 0 ? POSY : NEGY;

            const double dot_posterior_z =
                z_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation =
                dot_posterior_z > 0 ? z_coord_patient : -z_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_z > 0 ? POSZ : NEGZ;
        } else {
            _leftInfo.patient_orientation =
                dot_left_z > 0 ? z_coord_patient : -z_coord_patient;
            _leftInfo.volume_coord = dot_left_z > 0 ? POSZ : NEGZ;

            const double dot_posterior_y =
                y_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation =
                dot_posterior_y > 0 ? y_coord_patient : -y_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_y > 0 ? POSY : NEGY;
        }
    }
}

void CameraCalculator::calculate_vr_placement() {
    //视角方向和体的平面垂直，即不是和病人坐标系轴平行
    Vector3 view_dir = Vector3(0.0, 0.0, 0.0);

    if (_is_volume_orthogonal) {
        view_dir = _posteriorInfo.patient_orientation;
    } else {
        //对于斜扫数据的特殊处理，垂直于面 而非 平行于边
        view_dir = _headInfo.patient_orientation.cross_product(
                       _leftInfo.patient_orientation);
    }

    view_dir.normalize();

    Vector3 up = _headInfo.patient_orientation;

    double* spacing = _volume_data->_spacing;
    unsigned int* dim = _volume_data->_dim;
    const double max_length = std::max(
                                  std::max(dim[0] * spacing[0], dim[1] * spacing[1]), dim[2] * spacing[2]);

    Point3 eye = Point3(-view_dir.x, -view_dir.y, -view_dir.z) * max_length * 2;
    _vr_placement_camera.set_look_at(Point3(0.0, 0.0, 0.0));
    _vr_placement_camera.set_eye(eye);
    _vr_placement_camera.set_up_direction(up);
    _vr_placement_camera.set_ortho(-max_length * 0.5, max_length * 0.5,
                                   -max_length * 0.5, max_length * 0.5,
                                   -max_length * 2.0, max_length * 6.0);
}

void CameraCalculator::check_volume_orthogonal() {
    const Vector3& x_coord_patient = _volume_data->_image_orientation[0];
    const Vector3& y_coord_patient = _volume_data->_image_orientation[1];
    const Vector3& z_coord_patient = _volume_data->_image_orientation[2];

    double dot_xy = x_coord_patient.dot_product(y_coord_patient);
    double dot_xz = x_coord_patient.dot_product(z_coord_patient);
    double dot_yz = y_coord_patient.dot_product(z_coord_patient);

    if (abs(dot_xy) > DOUBLE_EPSILON || abs(dot_xz) > DOUBLE_EPSILON ||
            abs(dot_yz) > DOUBLE_EPSILON) {
        _is_volume_orthogonal = false;
    } else {
        _is_volume_orthogonal = true;
    }
}

bool CameraCalculator::check_volume_orthogonal() const {
    return _is_volume_orthogonal;
}

const Point3& CameraCalculator::get_default_mpr_center_world() const {
    return _default_mpr_center;
}

PatientAxisInfo CameraCalculator::get_head_patient_axis_info() const {
    return _headInfo;
}

PatientAxisInfo CameraCalculator::get_left_patient_axis_info() const {
    return _leftInfo;
}

PatientAxisInfo CameraCalculator::get_posterior_patient_axis_info() const {
    return _posteriorInfo;
}

ScanSliceType
CameraCalculator::check_scan_type(std::shared_ptr<OrthoCamera> camera) const {
    double spacing = 0;
    return check_scan_type(camera , spacing);
}

ScanSliceType CameraCalculator::check_scan_type(std::shared_ptr<OrthoCamera> camera , 
    double& spacing) const {
    Point3 eye = camera->get_eye();
    Point3 look_at = camera->get_look_at();
    Vector3 dir = look_at - eye;
    dir.normalize();

    const double dot_head = fabs(_headInfo.patient_orientation.dot_product(dir));
    const double dot_left = fabs(_leftInfo.patient_orientation.dot_product(dir));
    const double dot_posterior =
        fabs(_posteriorInfo.patient_orientation.dot_product(dir));

    ScanSliceType scan_type = OBLIQUE;
    spacing = 0;
    if (dot_head > dot_left && dot_head > dot_posterior) { // Transverse
        scan_type = fabs(dot_head - 1.0) > DOUBLE_EPSILON ? OBLIQUE : TRANSVERSE;
        spacing = _volume_data->_spacing[_headInfo.volume_coord/2];
    } else if (dot_left > dot_head && dot_left > dot_posterior) { // Sagittal
        scan_type = fabs(dot_left - 1.0) > DOUBLE_EPSILON ? OBLIQUE : SAGITTAL;
        spacing = _volume_data->_spacing[_leftInfo.volume_coord/2];
    } else { // Coronal
        scan_type = fabs(dot_posterior - 1.0) > DOUBLE_EPSILON ? OBLIQUE : CORONAL;
        spacing = _volume_data->_spacing[_posteriorInfo.volume_coord/2];
    }

    return scan_type;
}

bool CameraCalculator::page_orthogonal_mpr(
    std::shared_ptr<OrthoCamera> camera, int page_step , int& cur_page) const {
    double spacing_step = 0;
    ScanSliceType scan_type = check_scan_type(camera , spacing_step);
    if(scan_type == OBLIQUE){
        RENDERALGO_THROW_EXCEPTION("MPR is oblique!");
    }
    Point3 eye = camera->get_eye();
    Point3 look_at = camera->get_look_at();
    Vector3 dir = camera->get_view_direction();

    eye += dir * spacing_step * page_step;
    look_at += dir * spacing_step * page_step;

    const double distance =
        dir.dot_product(look_at - _othogonal_mpr_camera[scan_type].get_look_at());
    const double delta = distance / spacing_step;
    const int delta_i = int(delta);
    cur_page = get_default_page(scan_type) + delta_i;

    Point3 pt_test = _mat_world_to_volume.transform(look_at);
    //printf("mpr center volume : ");pt_test.print();
    //printf("\n");
    if (cur_page >= 0 && cur_page < get_page_maximum(scan_type)) {
        camera->set_eye(eye);
        camera->set_look_at(look_at);
        return true;
    } else {
        return false;
    }
}

bool CameraCalculator::page_orthogonal_mpr_to(
    std::shared_ptr<OrthoCamera> camera, int page) const {
    double spacing_step = 0;
    ScanSliceType scan_type = check_scan_type(camera , spacing_step);
    if(scan_type == OBLIQUE){
        RENDERALGO_THROW_EXCEPTION("MPR is oblique!");
    }
    Point3 eye = camera->get_eye();
    Point3 look_at = camera->get_look_at();
    Vector3 dir = camera->get_view_direction();

    // 2 page
    // 2.1 Back to default
    const Point3 ptOriCenter = get_default_mpr_center_world();
    const Point3 ptOriEye = _othogonal_mpr_camera[scan_type].get_eye();
    Point3 changed_look_at =
        look_at + (ptOriCenter - look_at).dot_product(dir) * dir;
    Point3 changed_eye = eye + (ptOriEye - eye).dot_product(dir) * dir;
    // 2.2 page to certain slice
    const int step = page - get_default_page(scan_type);
    changed_look_at += dir * (step) * spacing_step;
    changed_eye += dir * (step) * spacing_step;

    //const double distance = dir.dot_product(changed_look_at - _othogonal_mpr_camera[scan_type].get_look_at());
    //const double delta = distance / spacing_step;
    //const int delta_i = int(delta);
    //int cur_page = get_default_page(scan_type) + delta_i;

    if (page >= 0 && page < get_page_maximum(scan_type)) {
        camera->set_eye(changed_eye);
        camera->set_look_at(changed_look_at);
        return true;
    } else {
        return false;
    }
}

int CameraCalculator::get_orthogonal_mpr_page( std::shared_ptr<OrthoCamera> camera) const {
    int max_page;
    return get_orthogonal_mpr_page(camera, max_page);
}

int CameraCalculator::get_orthogonal_mpr_page(std::shared_ptr<OrthoCamera> camera, int& max_page) const {
    double spacing_step = 0;
    ScanSliceType scan_type = check_scan_type(camera , spacing_step);   
    if(scan_type == OBLIQUE){
        RENDERALGO_THROW_EXCEPTION("MPR is oblique!");
    }
    max_page = get_page_maximum(scan_type);
    const Point3 look_at = camera->get_look_at();
    const Vector3 dir = camera->get_view_direction();

    const double distance =
        dir.dot_product(look_at - _othogonal_mpr_camera[scan_type].get_look_at());
    const double delta = distance / spacing_step;
    const int delta_i = int(delta);
    const int page = get_default_page(scan_type) + delta_i;

    if (page >= 0 && page < get_page_maximum(scan_type)) {
        return page;
    } else {
        RENDERALGO_THROW_EXCEPTION("Calculate MPR page failed!");
    }
}

void CameraCalculator::translate_mpr_to(std::shared_ptr<OrthoCamera> camera,
                                        const Point3& pt) {
    const Point3 look_at = camera->get_look_at();
    const Point3 eye = camera->get_eye();
    const Vector3 dir = camera->get_view_direction();

    const double translate = (pt - look_at).dot_product(dir);
    camera->set_look_at(look_at + translate * dir);
    camera->set_eye(eye + translate * dir);
}

// float
// CameraCalculator::convert_thickness_world_to_volume(std::shared_ptr<OrthoCamera>
// pMPRCamera , float fThicknessWorldmm)const
//{
//    Vector3 dir = (pMPRCamera->get_look_at() -
//    pMPRCamera->get_eye()).get_normalize()*fThicknessWorldmm;
//    dir = m_matVolume2Wolrd.get_transpose().transform(dir);
//    return (float)(dir.magnitude());
//}

int CameraCalculator::get_page_maximum(ScanSliceType type) const {
    switch (type) {
    case TRANSVERSE: {
        return _volume_data->_dim[_headInfo.volume_coord / 2];
    }

    case SAGITTAL: {
        return _volume_data->_dim[_leftInfo.volume_coord / 2];
    }

    case CORONAL: {
        return _volume_data->_dim[_posteriorInfo.volume_coord / 2];
    }

    default: {
        RENDERALGO_THROW_EXCEPTION("Cant get oblique maximun page!");
    }
    }
}

int CameraCalculator::get_default_page(ScanSliceType type) const {
    switch (type) {
    case TRANSVERSE: {
        const int idx = _headInfo.volume_coord / 2;
        Vector3 vl = _volume_data->_image_orientation[idx] - Vector3(0.0,0.0,1.0);
        Vector3 vr = _volume_data->_image_orientation[idx] - Vector3(0.0,0.0,-1.0);
        if (vl.magnitude() < vr.magnitude()) {
            return _volume_data->_dim[idx] / 2;
        } else {
            //paging direction is inverse with volume direction
            return _volume_data->_dim[idx] / 2 - 1;
        }
    }

    case SAGITTAL: {//This paging direction is not the positive direction of patient coordinate
        const int idx = _leftInfo.volume_coord / 2;
        Vector3 vl = _volume_data->_image_orientation[idx] - Vector3(1.0,0.0,0.0);
        Vector3 vr = _volume_data->_image_orientation[idx] - Vector3(-1.0,0.0,0.0);
        if (vl.magnitude() < vr.magnitude()) {
            //paging direction is inverse with volume direction
            return _volume_data->_dim[idx] / 2 - 1;
        } else {
            return _volume_data->_dim[idx] / 2;
        }
    }

    case CORONAL: {
        const int idx = _posteriorInfo.volume_coord / 2;
        Vector3 vl = _volume_data->_image_orientation[idx] - Vector3(0.0,1.0,0.0);
        Vector3 vr = _volume_data->_image_orientation[idx] - Vector3(0.0,-1.0,0.0);
        if (vl.magnitude() < vr.magnitude()) {
            return _volume_data->_dim[idx] / 2;
        } else {
            //paging direction is inverse with volume direction
            return _volume_data->_dim[idx] / 2 - 1;
        }
    }

    default: {
        RENDERALGO_THROW_EXCEPTION("Cant get oblique default page!");
    }
    }
}

void CameraCalculator::caluculate_orthogonal_mpr_placement() {
    double* spacing = _volume_data->_spacing;
    unsigned int* dim = _volume_data->_dim;
    const double max_length = std::max(
                                  std::max(dim[0] * spacing[0], dim[1] * spacing[1]), dim[2] * spacing[2]);

    const Point3 look_at = _default_mpr_center;
    Vector3 up;
    Point3 eye;
    ScanSliceType scan_type[3] = {SAGITTAL, CORONAL, TRANSVERSE};

    for (int i = 0; i < 3; ++i) {
        switch (scan_type[i]) {
        case SAGITTAL: {
            Vector3 vecx = Vector3(0.0, 0.0, 0.0);

            if (_is_volume_orthogonal) {
                vecx = _leftInfo.patient_orientation;
            } else {
                vecx = _posteriorInfo.patient_orientation.cross_product(
                           _headInfo.patient_orientation);
            }

            vecx.normalize();

            eye = look_at + vecx * max_length * 2;
            up = Vector3(_headInfo.patient_orientation);
            break;
        }

        case TRANSVERSE: {
            Vector3 vecz = Vector3(0.0, 0.0, 0.0);

            if (_is_volume_orthogonal) {
                vecz = -_headInfo.patient_orientation;
            } else {
                vecz = -_leftInfo.patient_orientation.cross_product(
                           _posteriorInfo.patient_orientation);
            }

            vecz.normalize();

            eye = look_at + vecz * max_length * 2;
            up = -Vector3(_posteriorInfo.patient_orientation);
            break;
        }

        case CORONAL: {
            Vector3 vecy = Vector3(0.0, 0.0, 0.0);

            if (_is_volume_orthogonal) {
                vecy = -_posteriorInfo.patient_orientation;
            } else {
                vecy = -_headInfo.patient_orientation.cross_product(
                           _leftInfo.patient_orientation);
            }

            vecy.normalize();

            eye = look_at + vecy * max_length * 2;
            up = Vector3(_headInfo.patient_orientation);
            break;
        }

        default: {
            RENDERALGO_THROW_EXCEPTION("Invalid scan slice type!");
        }
        }

        _othogonal_mpr_camera[i].set_look_at(look_at);
        _othogonal_mpr_camera[i].set_eye(eye);
        _othogonal_mpr_camera[i].set_up_direction(up);
        _othogonal_mpr_camera[i].set_ortho(-max_length * 0.5, max_length * 0.5,
                                           -max_length * 0.5, max_length * 0.5,
                                           -max_length * 2.0, max_length * 6.0);
        _othogonal_mpr_norm[i] = Vector3(look_at - eye).get_normalize();
    }
}

void CameraCalculator::calculate_default_mpr_center_world() {
    Point3 look_at = Point3(0.0, 0.0, 0.0);
    // Sagittal translate
    Vector3 view_dir = _leftInfo.patient_orientation;
    view_dir.normalize();
    const unsigned int uiSagittalDimension =
        _volume_data->_dim[_leftInfo.volume_coord / 2];

    if (uiSagittalDimension % 2 != 0) {
        look_at -= view_dir * 0.5 *
                   _volume_data->_spacing[_leftInfo.volume_coord / 2]; //floor
    }

    // Transversal translate
    view_dir = _headInfo.patient_orientation;
    view_dir.normalize();
    const unsigned int uiTransversalDimension =
        _volume_data->_dim[_headInfo.volume_coord / 2];

    if (uiTransversalDimension % 2 != 0) {
        look_at -= view_dir * 0.5 *
                   _volume_data->_spacing[_headInfo.volume_coord / 2]; //floor
    }

    // Coronal translate
    view_dir = _posteriorInfo.patient_orientation;
    view_dir.normalize();
    const unsigned int uiCoronalDimension =
        _volume_data->_dim[_posteriorInfo.volume_coord / 2];

    if (uiCoronalDimension % 2 != 0) {
        look_at -= view_dir * 0.5 *
                   _volume_data->_spacing[_posteriorInfo.volume_coord / 2]; //floor
    }

    _default_mpr_center = look_at;
    Point3 pt_test = _mat_volume_to_wolrd.get_inverse().transform(_default_mpr_center);
    MI_RENDERALGO_LOG(MI_DEBUG) << "MPR volume center: " << pt_test;
}

const Matrix4& CameraCalculator::get_patient_to_world_matrix() const {
    return _mat_patient_to_world;
}

// Point3 CameraCalculator::adjust_point_to_discrete(const Point3& ptWorld)
// const
//{
//    Point3 ptVolume = m_matWorld2Volume.transform(ptWorld);
//    ptVolume.x = (double)( (int)ptVolume.x);
//    ptVolume.y = (double)( (int)ptVolume.y);
//    ptVolume.z = (double)( (int)ptVolume.z);
//
//    return m_matVolume2Wolrd.transform(ptVolume);
//}

MED_IMG_END_NAMESPACE