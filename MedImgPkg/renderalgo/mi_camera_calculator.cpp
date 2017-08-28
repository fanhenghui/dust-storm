#include "mi_camera_calculator.h"
#include "io/mi_image_data.h"
#include "arithmetic/mi_arithmetic_utils.h"

MED_IMG_BEGIN_NAMESPACE 

CameraCalculator::CameraCalculator(std::shared_ptr<ImageData> image_data):_volume_data(image_data),_is_volume_orthogonal(true)
{
    calculate_i();
}
 
//void CameraCalculator::UpdateImageData(std::shared_ptr<ImageData> image_data)
//{
//    _volume_data = image_data;
//    Calculate_i();
//}

const Matrix4& CameraCalculator::get_volume_to_physical_matrix()const
{
    
    return _mat_volume_to_physical;
}

const Matrix4& CameraCalculator::get_physical_to_world_matrix()const
{
    return _mat_physical_to_world;
}


const Matrix4& CameraCalculator::get_world_to_patient_matrix()const
{
    return _mat_world_to_patient;
}

const Matrix4& CameraCalculator::get_volume_to_world_matrix()const
{
    return _mat_volume_to_wolrd;
}

const Matrix4& CameraCalculator::get_world_to_volume_matrix()const
{
    return _mat_world_to_volume;
}

void CameraCalculator::init_mpr_placement(std::shared_ptr<OrthoCamera> camera , ScanSliceType type , const Point3& ptCenterPoint)const
{
    *camera = _othogonal_mpr_camera[static_cast<int>(type)];
    camera->set_look_at(ptCenterPoint);
}

void CameraCalculator::init_mpr_placement(std::shared_ptr<OrthoCamera> camera , ScanSliceType type)const
{
    *camera = _othogonal_mpr_camera[static_cast<int>(type)];
}

void CameraCalculator::init_vr_placement(std::shared_ptr<OrthoCamera> camera)const
{
    *camera = _vr_placement_camera;
}

void CameraCalculator::calculate_i()
{
    //Check orthogonal ����CT������б��������˵����ɨ���X��Y���򲢲���������Ⱦ��Ҫ���⴦��
    check_volume_orthogonal_i();

    //���������ݵ�ÿ����ͱ�׼��������ϵ�µ���Ĺ�ϵ
    calculate_patient_axis_info_i();

    //Calculate volume to physical/world/patient
    calculate_matrix_i();

    //Calculate VR replacement
    calculate_vr_placement_i();

    //Calculate orthogonal MPR replacement
    calculate_default_mpr_center_world_i();
    caluculate_orthogonal_mpr_placement_i();
}

void CameraCalculator::calculate_matrix_i()
{
    //1 Calculate volume to physical
    unsigned int *dim = _volume_data->_dim;
    double *spacing = _volume_data->_spacing;

    _mat_volume_to_physical.set_idintity();
    _mat_volume_to_physical.prepend(make_translate(-Vector3(dim[0] * 0.5, dim[1] * 0.5, dim[2] * 0.5)));
    _mat_volume_to_physical.prepend(make_scale(Vector3(spacing[0], spacing[1], spacing[2])));

    //2 Calculate physical to world
    //MatA2B * PA = PB; -----> MatA2B = PB*Inv(PA);
    Vector3 standard_coord_axis[6] = {
        Vector3(1.0,0.0,0.0),
        Vector3(-1.0,0.0,0.0),
        Vector3(0.0,1.0,0.0),
        Vector3(0.0,-1.0,0.0),
        Vector3(0.0,0.0,1.0),
        Vector3(0.0,0.0,-1.0)
    };

    const Vector3 physical_head = standard_coord_axis[(int)(_headInfo.volume_coord)];
    const Vector3 physical_posterior = standard_coord_axis[(int)(_posteriorInfo.volume_coord)];
    const Vector3 physical_left = standard_coord_axis[(int)(_leftInfo.volume_coord)];

    const Matrix4 mat_physical(physical_head.x,physical_head.y,physical_head.z,0.0,
        physical_posterior.x, physical_posterior.y, physical_posterior.z,0.0,
        physical_left.x, physical_left.y, physical_left.z,0.0,
        0.0,0.0,0.0,1.0);

    const Vector3 patient_head = _headInfo.patient_orientation;
    const Vector3 patient_posterior = _posteriorInfo.patient_orientation;
    const Vector3 patient_left = _leftInfo.patient_orientation;

    const Matrix4 mat_patient(patient_head.x,patient_head.y,patient_head.z,0.0,
        patient_posterior.x, patient_posterior.y, patient_posterior.z,0.0,
        patient_left.x, patient_left.y, patient_left.z,0.0,
        0.0,0.0,0.0,1.0);

    if (!mat_physical.has_inverse())
    {
        _mat_physical_to_world.set_idintity();
    }
    else
    {
        _mat_physical_to_world = mat_patient * mat_physical.get_inverse();
    }

    //3 Calculate volume to world
    _mat_volume_to_wolrd = _mat_physical_to_world*_mat_volume_to_physical;
    _mat_world_to_volume = _mat_volume_to_wolrd.get_inverse();

    //4 Calculate world to patient
    const Point3 &img_position = _volume_data->_image_position;
    const Point3 &img_world = _mat_volume_to_wolrd.transform(Point3::S_ZERO_POINT);
    _mat_world_to_patient = make_translate(img_position - img_world);
    _mat_patient_to_world = _mat_world_to_patient.get_inverse();
}


void CameraCalculator::calculate_patient_axis_info_i()
{
    const Vector3& x_coord_patient = _volume_data->_image_orientation[0];
    const Vector3& y_coord_patient = _volume_data->_image_orientation[1];
    const Vector3& z_coord_patient = _volume_data->_image_orientation[2];

    const double *spacing = _volume_data->_spacing;
    const unsigned int *dim = _volume_data->_dim;

    /// rotate the volume to get consistent with patient coordinate 
    const Vector3 standard_haed_axis(0.0,0.0,1.0);
    const Vector3 standard_left_axis(1.0,0.0,0.0);
    const Vector3 standard_posterior_axis(0.0,1.0,0.0);

    const double dot_head_x = x_coord_patient.dot_product(standard_haed_axis);
    const double dot_head_y = y_coord_patient.dot_product(standard_haed_axis);
    const double dot_head_z = z_coord_patient.dot_product(standard_haed_axis);

    /// Priority is TRA>COR>SAG.
    if ((!(abs(dot_head_z) - abs(dot_head_x)<0)) && (!(abs(dot_head_z) - abs(dot_head_y)<0)))
    {
        _headInfo.patient_orientation = dot_head_z > 0 ? z_coord_patient : -z_coord_patient;
        _headInfo.volume_coord = dot_head_z > 0 ? POSZ : NEGZ;

        const double dot_left_x = x_coord_patient.dot_product(standard_left_axis);
        const double dot_left_y = y_coord_patient.dot_product(standard_left_axis);

        if (!(abs(dot_left_x) < abs(dot_left_y)))
        {
            _leftInfo.patient_orientation = dot_left_x > 0 ? x_coord_patient : -x_coord_patient;
            _leftInfo.volume_coord = dot_left_x > 0 ? POSX : NEGX;

            const double dot_posterior_y = y_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation = dot_posterior_y > 0 ? y_coord_patient : -y_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_y > 0 ? POSY : NEGY;
        }
        else
        {
            _leftInfo.patient_orientation = dot_left_y > 0 ? y_coord_patient : -y_coord_patient;
            _leftInfo.volume_coord = dot_left_y > 0 ? POSY : NEGY;

            const double dot_posterior_x = x_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation = dot_posterior_x > 0 ? x_coord_patient : -x_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_x > 0 ? POSX : NEGX;
        }

    }
    else if ((!(abs(dot_head_y) - abs(dot_head_x) < 0))&&(!(abs(dot_head_y) - abs(dot_head_z) < 0)))
    {
        _headInfo.patient_orientation = dot_head_y > 0? y_coord_patient : -y_coord_patient;
        _headInfo.volume_coord = dot_head_y > 0? POSY : NEGY;

        const double dot_left_x  = x_coord_patient.dot_product(standard_left_axis);
        const double dot_left_z  = z_coord_patient.dot_product(standard_left_axis);

        if (!(abs(dot_left_x) < abs(dot_left_z)))
        {
            _leftInfo.patient_orientation = dot_left_x > 0 ? x_coord_patient : -x_coord_patient;
            _leftInfo.volume_coord = dot_left_x > 0 ? POSX : NEGX;

            const double dot_posterior_z = z_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation = dot_posterior_z > 0? z_coord_patient : -z_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_z > 0 ? POSZ : NEGZ;
        }
        else
        {
            _leftInfo.patient_orientation = dot_left_z > 0 ? z_coord_patient : -z_coord_patient;
            _leftInfo.volume_coord = dot_left_z > 0 ? POSZ : NEGZ;

            const double dot_posterior_x = x_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation = dot_posterior_x > 0? x_coord_patient : -x_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_x > 0 ? POSX : NEGX;
        }

    }
    else
    {
        _headInfo.patient_orientation = dot_head_x > 0? x_coord_patient : -x_coord_patient;
        _headInfo.volume_coord = dot_head_x > 0 ? POSX : NEGX;

        const double dot_left_y  = y_coord_patient.dot_product(standard_left_axis);
        const double dot_left_z  = z_coord_patient.dot_product(standard_left_axis);

        if (!(abs(dot_left_y) < abs(dot_left_z)))
        {
            _leftInfo.patient_orientation = dot_left_y > 0? y_coord_patient : -y_coord_patient;
            _leftInfo.volume_coord = dot_left_y > 0 ? POSY : NEGY;

            const double dot_posterior_z = z_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation = dot_posterior_z > 0? z_coord_patient : -z_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_z > 0 ? POSZ : NEGZ;
        }
        else
        {
            _leftInfo.patient_orientation = dot_left_z > 0 ? z_coord_patient : -z_coord_patient;
            _leftInfo.volume_coord = dot_left_z > 0 ? POSZ : NEGZ;

            const double dot_posterior_y = y_coord_patient.dot_product(standard_posterior_axis);
            _posteriorInfo.patient_orientation = dot_posterior_y > 0 ? y_coord_patient : -y_coord_patient;
            _posteriorInfo.volume_coord = dot_posterior_y > 0 ? POSY : NEGY;
        }
    }
}

void CameraCalculator::calculate_vr_placement_i()
{
    //�ӽǷ�������ƽ�洹ֱ�������ǺͲ�������ϵ��ƽ��
    Vector3 view_dir = Vector3(0.0, 0.0, 0.0);
    
    if (_is_volume_orthogonal)
    {
        view_dir = _posteriorInfo.patient_orientation;
    }
    else//����бɨ���ݵ����⴦����ֱ���� ���� ƽ���ڱ�
    {
        view_dir = _headInfo.patient_orientation.cross_product(_leftInfo.patient_orientation);
    }
    view_dir.normalize();

    Vector3 up = _headInfo.patient_orientation;


    double *spacing = _volume_data->_spacing;
    unsigned int *dim = _volume_data->_dim;
    const double max_length = std::max(std::max(dim[0] * spacing[0],dim[1] * spacing[1]),dim[2] * spacing[2]);

    Point3 eye = Point3(-view_dir.x, -view_dir.y, -view_dir.z)*max_length*2;
    _vr_placement_camera.set_look_at(Point3(0.0,0.0,0.0));
    _vr_placement_camera.set_eye(eye);
    _vr_placement_camera.set_up_direction(up);
    _vr_placement_camera.set_ortho(-max_length*0.5 , max_length*0.5 , -max_length*0.5 , max_length*0.5 , -max_length*2.0 , max_length*6.0);
}

void CameraCalculator::check_volume_orthogonal_i()
{
    const Vector3& x_coord_patient = _volume_data->_image_orientation[0];
    const Vector3& y_coord_patient = _volume_data->_image_orientation[1];
    const Vector3& z_coord_patient = _volume_data->_image_orientation[2];

    double dot_xy = x_coord_patient.dot_product(y_coord_patient);
    double dot_xz = x_coord_patient.dot_product(z_coord_patient);
    double dot_yz = y_coord_patient.dot_product(z_coord_patient);

    if ( abs(dot_xy) > DOUBLE_EPSILON || abs(dot_xz) > DOUBLE_EPSILON || abs(dot_yz) > DOUBLE_EPSILON )
    {
        _is_volume_orthogonal =  false;
    }
    else
    {
        _is_volume_orthogonal = true;
    }
}

bool CameraCalculator::check_volume_orthogonal()const
{
    return _is_volume_orthogonal;
}

const Point3& CameraCalculator::get_default_mpr_center_world()const
{
    return _default_mpr_center;
}

PatientAxisInfo CameraCalculator::get_head_patient_axis_info()const
{
    return _headInfo;
}

PatientAxisInfo CameraCalculator::get_left_patient_axis_info()const
{
    return _leftInfo;
}

PatientAxisInfo CameraCalculator::get_posterior_patient_axis_info()const
{
    return _posteriorInfo;
}

ScanSliceType CameraCalculator::check_scan_type(std::shared_ptr<OrthoCamera> camera)const
{
    Point3 eye = camera->get_eye();
    Point3 look_at = camera->get_look_at();
    Vector3 dir = look_at - eye;
    dir.normalize();

    const double dot_head = fabs(_headInfo.patient_orientation.dot_product(dir));
    const double dot_left = fabs(_leftInfo.patient_orientation.dot_product(dir));
    const double dot_posterior = fabs(_posteriorInfo.patient_orientation.dot_product(dir));

    ScanSliceType scan_type = OBLIQUE;
    if (dot_head > dot_left && dot_head > dot_posterior)//Transverse
    {
        scan_type = fabs(dot_head - 1.0) > DOUBLE_EPSILON ? OBLIQUE : TRANSVERSE;
    }
    else if (dot_left > dot_head && dot_left > dot_posterior)//Sagittal
    {
        scan_type = fabs(dot_left - 1.0) > DOUBLE_EPSILON ? OBLIQUE : SAGITTAL;
    }
    else//Coronal
    {
        scan_type = fabs(dot_posterior - 1.0) > DOUBLE_EPSILON ? OBLIQUE : CORONAL;
    }
    return scan_type;
}

bool CameraCalculator::page_orthognal_mpr(std::shared_ptr<OrthoCamera> camera , int iPageStep)const
{
    Point3 eye = camera->get_eye();
    Point3 look_at = camera->get_look_at();
    const Vector3 dir = camera->get_view_direction();

    const double dot_head = fabs(_headInfo.patient_orientation.dot_product(dir));
    const double dot_left = fabs(_leftInfo.patient_orientation.dot_product(dir));
    const double dot_posterior = fabs(_posteriorInfo.patient_orientation.dot_product(dir));

    double spacing_step = 0;
    if (dot_head > dot_left && dot_head > dot_posterior)//Transverse
    {
        spacing_step = _volume_data->_spacing[_headInfo.volume_coord/2];
    }
    else if (dot_left > dot_head && dot_left > dot_posterior)//Sagittal
    {
        spacing_step = _volume_data->_spacing[_leftInfo.volume_coord/2];
    }
    else//Coronal
    {
        spacing_step = _volume_data->_spacing[_posteriorInfo.volume_coord/2];
    }

    eye += dir*spacing_step*iPageStep;
    look_at+= dir*spacing_step*iPageStep;
    Point3 pt_v = _mat_world_to_volume.transform(look_at);
    if (ArithmeticUtils::check_in_bound(pt_v , Point3(_volume_data->_dim[0]-1 , _volume_data->_dim[1]-1 , _volume_data->_dim[2]-1)))
    {
        camera->set_eye(eye);
        camera->set_look_at(look_at);
        return true;
    }
    else
    {
        return false;
    }
}

bool CameraCalculator::page_orthognal_mpr_to(std::shared_ptr<OrthoCamera> camera , int page)const
{
    //1 Check orthogonal
    const Point3 eye = camera->get_eye();
    const Point3 look_at = camera->get_look_at();
    const Vector3 dir = camera->get_view_direction();

    const double dot_head = fabs(_headInfo.patient_orientation.dot_product(dir));
    const double dot_left = fabs(_leftInfo.patient_orientation.dot_product(dir));
    const double dot_posterior = fabs(_posteriorInfo.patient_orientation.dot_product(dir));

    ScanSliceType scan_type = OBLIQUE;
    double spacing_step = 0;
    if (dot_head > dot_left && dot_head > dot_posterior)//Transverse
    {
        scan_type = fabs(dot_head - 1.0) > DOUBLE_EPSILON ? OBLIQUE : TRANSVERSE;
        spacing_step = _volume_data->_spacing[_headInfo.volume_coord/2];
    }
    else if (dot_left > dot_head && dot_left > dot_posterior)//Sagittal
    {
        scan_type = fabs(dot_left - 1.0) > DOUBLE_EPSILON ? OBLIQUE : SAGITTAL;
        spacing_step = _volume_data->_spacing[_leftInfo.volume_coord/2];
    }
    else//Coronal
    {
        scan_type = fabs(dot_posterior - 1.0) > DOUBLE_EPSILON ? OBLIQUE : CORONAL;
        spacing_step = _volume_data->_spacing[_posteriorInfo.volume_coord/2];
    }

    if (scan_type == OBLIQUE)
    {
        return false;
    }

    //2 page
    //2.1 Back to default
    const Point3 ptOriCenter = get_default_mpr_center_world();
    const Point3 ptOriEye = _othogonal_mpr_camera[scan_type].get_eye();
    Point3 changed_look_at = look_at + (ptOriCenter - look_at).dot_product(dir) * dir;
    Point3 changed_eye = eye + (ptOriEye - eye).dot_product(dir) * dir;
    //2.2 page to certain slice
    const int step = page - get_default_page(scan_type);
    changed_look_at += dir* (step ) * spacing_step;
    changed_eye += dir* (step ) * spacing_step;

    Point3 ptV = _mat_world_to_volume.transform(changed_look_at);
    if (ArithmeticUtils::check_in_bound(ptV , Point3(_volume_data->_dim[0]-1 , _volume_data->_dim[1]-1 , _volume_data->_dim[2]-1)))
    {
        camera->set_eye(changed_eye);
        camera->set_look_at(changed_look_at);
        return true;
    }
    else
    {
        return false;
    }
}

int CameraCalculator::get_orthognal_mpr_page(std::shared_ptr<OrthoCamera> camera) const
{
    //1 Check orthogonal
    const Point3 eye = camera->get_eye();
    const Point3 look_at = camera->get_look_at();
    const Vector3 dir = camera->get_view_direction();

    const double dot_head = fabs(_headInfo.patient_orientation.dot_product(dir));
    const double dot_left = fabs(_leftInfo.patient_orientation.dot_product(dir));
    const double dot_posterior = fabs(_posteriorInfo.patient_orientation.dot_product(dir));

    ScanSliceType scan_type = OBLIQUE;
    double spacing_step = 0;
    if (dot_head > dot_left && dot_head > dot_posterior)//Transverse
    {
        scan_type = fabs(dot_head - 1.0) > DOUBLE_EPSILON ? OBLIQUE : TRANSVERSE;
        spacing_step = _volume_data->_spacing[_headInfo.volume_coord/2];
    }
    else if (dot_left > dot_head && dot_left > dot_posterior)//Sagittal
    {
        scan_type = fabs(dot_left - 1.0) > DOUBLE_EPSILON ? OBLIQUE : SAGITTAL;
        spacing_step = _volume_data->_spacing[_leftInfo.volume_coord/2];
    }
    else//Coronal
    {
        scan_type = fabs(dot_posterior - 1.0) > DOUBLE_EPSILON ? OBLIQUE : CORONAL;
        spacing_step = _volume_data->_spacing[_posteriorInfo.volume_coord/2];
    }

    if (scan_type == OBLIQUE)
    {
        RENDERALGO_THROW_EXCEPTION("Calculate MPR page failed!");
    }

    const double distance = dir.dot_product(look_at - _othogonal_mpr_camera[scan_type].get_look_at());
    int iDelta = int(distance/spacing_step);
    int page = get_default_page(scan_type) + iDelta;
    if (page >= 0 && page < get_page_maximum(scan_type))
    {
        return page;
    }
    else
    {
        RENDERALGO_THROW_EXCEPTION("Calculate MPR page failed!");
    }

}

void CameraCalculator::translate_mpr_to(std::shared_ptr<OrthoCamera> camera , const Point3& pt)
{
    const Point3 look_at = camera->get_look_at();
    const Point3 eye = camera->get_eye();
    const Vector3 dir = camera->get_view_direction();

    const double translate = (pt - look_at).dot_product(dir);
    camera->set_look_at(look_at + translate* dir);
    camera->set_eye(eye + translate* dir);
}

//float CameraCalculator::convert_thickness_world_to_volume(std::shared_ptr<OrthoCamera> pMPRCamera , float fThicknessWorldmm)const
//{
//    Vector3 dir = (pMPRCamera->get_look_at() - pMPRCamera->get_eye()).get_normalize()*fThicknessWorldmm;
//    dir = m_matVolume2Wolrd.get_transpose().transform(dir);
//    return (float)(dir.magnitude());
//}

int CameraCalculator::get_page_maximum(ScanSliceType type) const
{
    switch(type)
    {
    case TRANSVERSE:
        {
            return _volume_data->_dim[_headInfo.volume_coord/2];
        }
    case SAGITTAL:
        {
            return _volume_data->_dim[_leftInfo.volume_coord/2];
        }
    case CORONAL:
        {
            return _volume_data->_dim[_posteriorInfo.volume_coord/2];
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Cant get oblique maximun page!");
        }
    }
}

int CameraCalculator::get_default_page(ScanSliceType type) const
{
    switch(type)
    {
    case TRANSVERSE:
        {
            return _volume_data->_dim[_headInfo.volume_coord/2]/2;
        }
    case SAGITTAL:
        {
            return _volume_data->_dim[_leftInfo.volume_coord/2]/2;
        }
    case CORONAL:
        {
            return _volume_data->_dim[_posteriorInfo.volume_coord/2]/2;
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Cant get oblique default page!");
        }
    }
}

void CameraCalculator::caluculate_orthogonal_mpr_placement_i()
{
    double *spacing = _volume_data->_spacing;
    unsigned int *dim = _volume_data->_dim;
    const double max_length = std::max(std::max(dim[0] * spacing[0],dim[1] * spacing[1]),dim[2] * spacing[2]);

    const Point3 look_at = _default_mpr_center;
    Vector3 up;
    Point3 eye;
    ScanSliceType scan_type[3] = {SAGITTAL , CORONAL, TRANSVERSE};
    for (int i = 0 ; i< 3 ; ++i)
    {
        switch(scan_type[i])
        {
        case SAGITTAL:
            {
                Vector3 vecx = Vector3(0.0, 0.0, 0.0);
                if (_is_volume_orthogonal)
                {
                    vecx = _leftInfo.patient_orientation;
                }
                else
                {
                    vecx = _posteriorInfo.patient_orientation.cross_product(_headInfo.patient_orientation);
                }
                vecx.normalize();

                eye = look_at + vecx*max_length*2;
                up = Vector3(_headInfo.patient_orientation);
                break;
            }

        case TRANSVERSE:
            {
                Vector3 vecz = Vector3(0.0, 0.0, 0.0);
                if (_is_volume_orthogonal)
                {
                    vecz = -_headInfo.patient_orientation;
                }
                else
                {
                    vecz = -_leftInfo.patient_orientation.cross_product(_posteriorInfo.patient_orientation);
                }
                vecz.normalize();

                eye = look_at + vecz*max_length*2;
                up = -Vector3(_posteriorInfo.patient_orientation);
                break;
            }
        case CORONAL:
            {
                Vector3 vecy = Vector3(0.0, 0.0, 0.0);
                if (_is_volume_orthogonal)
                {
                    vecy = -_posteriorInfo.patient_orientation;
                }
                else
                {
                    vecy = -_headInfo.patient_orientation.cross_product(_leftInfo.patient_orientation);
                }

                vecy.normalize();

                eye = look_at + vecy*max_length*2;
                up = Vector3(_headInfo.patient_orientation);
                break;
            }
        default:
            {
                RENDERALGO_THROW_EXCEPTION("Invalid scan slice type!");
            }
        }

        _othogonal_mpr_camera[i].set_look_at(look_at);
        _othogonal_mpr_camera[i].set_eye(eye);
        _othogonal_mpr_camera[i].set_up_direction(up);
        _othogonal_mpr_camera[i].set_ortho(-max_length*0.5 , max_length*0.5 , -max_length*0.5 , max_length*0.5 , -max_length*2.0 , max_length*6.0);
        _othogonal_mpr_norm[i] = Vector3(look_at - eye).get_normalize();
    }
}

void CameraCalculator::calculate_default_mpr_center_world_i()
{
    Point3 look_at = Point3(0.0, 0.0, 0.0);
    //Sagittal translate
    Vector3 view_dir = _leftInfo.patient_orientation;
    view_dir.normalize();
    const unsigned int uiSagittalDimension = _volume_data->_dim[_leftInfo.volume_coord/2];
    if ( uiSagittalDimension % 2 == 0 )
    {
        look_at -= view_dir*0.5*_volume_data->_spacing[_leftInfo.volume_coord/2];//ȡ����
    }

    //Transversal translate
    view_dir = _headInfo.patient_orientation;
    view_dir.normalize();
    const unsigned int uiTransversalDimension = _volume_data->_dim[_headInfo.volume_coord/2];
    if ( 0 == uiTransversalDimension % 2)
    {
        look_at -= view_dir*0.5*_volume_data->_spacing[_headInfo.volume_coord/2];//ȡ����
    }

    //Coronal translate
    view_dir = _posteriorInfo.patient_orientation;
    view_dir.normalize();
    const unsigned int uiCoronalDimension = _volume_data->_dim[_posteriorInfo.volume_coord/2];
    if ( 0 == uiCoronalDimension % 2)
    {
        look_at -= view_dir*0.5*_volume_data->_spacing[_posteriorInfo.volume_coord/2];//ȡ����
    }

    _default_mpr_center = look_at;
}

const Matrix4& CameraCalculator::get_patient_to_world_matrix() const
{
    return _mat_patient_to_world;
}

//Point3 CameraCalculator::adjust_point_to_discrete(const Point3& ptWorld) const
//{
//    Point3 ptVolume = m_matWorld2Volume.transform(ptWorld);
//    ptVolume.x = (double)( (int)ptVolume.x);
//    ptVolume.y = (double)( (int)ptVolume.y);
//    ptVolume.z = (double)( (int)ptVolume.z);
//
//    return m_matVolume2Wolrd.transform(ptVolume);
//}






MED_IMG_END_NAMESPACE