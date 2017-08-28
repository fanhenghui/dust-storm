#include "mi_observer_voi_statistic.h"

#include "arithmetic/mi_volume_statistician.h"
#include "io/mi_image_data.h"

#include "renderalgo/mi_camera_calculator.h"
#include "renderalgo/mi_volume_infos.h"


#include "mi_model_voi.h"

MED_IMG_BEGIN_NAMESPACE 

VOIStatisticObserver::VOIStatisticObserver()
{

}

VOIStatisticObserver::~VOIStatisticObserver()
{

}

void VOIStatisticObserver::set_model(std::shared_ptr<VOIModel> model)
{
    _model = model;
}

void VOIStatisticObserver::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos)
{
    _volume_infos = volume_infos;
}

void VOIStatisticObserver::update(int code_id)
{
    if(1 == code_id)
    {
        return;
    }

    QTWIDGETS_CHECK_NULL_EXCEPTION(_volume_infos);

    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    QTWIDGETS_CHECK_NULL_EXCEPTION(volume_data);

    std::shared_ptr<ImageData> mask_data = _volume_infos->get_mask();
    QTWIDGETS_CHECK_NULL_EXCEPTION(mask_data);

    std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();
    QTWIDGETS_CHECK_NULL_EXCEPTION(camera_cal);

    std::shared_ptr<VOIModel> model = _model.lock();
    QTWIDGETS_CHECK_NULL_EXCEPTION(model);

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

    const int voi_num = model->get_voi_number();
    unsigned int pixel_num;
    double min , max , mean , var , std;
    for (int i = 0; i<voi_num ; ++i)
    {
        const VOISphere voi = model->get_voi(i);
        const unsigned char label = model->get_label(i);

        Ellipsoid ellipsoid;
        ellipsoid._center = mat_p2v.transform(voi.center);
        double voi_abc[3] = {0,0,0};
        voi_abc[head_info.volume_coord/2] = voi.diameter*0.5/basic_abc[head_info.volume_coord/2] ;
        voi_abc[left_info.volume_coord/2] = voi.diameter*0.5/basic_abc[left_info.volume_coord/2] ;
        voi_abc[posterior_info.volume_coord/2] = voi.diameter*0.5/basic_abc[posterior_info.volume_coord/2] ;
        ellipsoid._a = voi_abc[0];
        ellipsoid._b = voi_abc[1];
        ellipsoid._c = voi_abc[2];

        switch(volume_data->_data_type)
        {
        case SHORT:
            {
                VolumeStatistician<short> sta;
                sta.set_data_ref((short*)volume_data->get_pixel_pointer());
                sta.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
                sta.set_dim(volume_data->_dim);
                sta.set_target_labels(std::vector<unsigned char>(1, label));
                sta.get_intensity_analysis(ellipsoid, pixel_num , min , max , mean , var , std);

                break;
            }
        case USHORT:
            {
                VolumeStatistician<unsigned short> sta;
                sta.set_data_ref((unsigned short*)volume_data->get_pixel_pointer());
                sta.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
                sta.set_dim(volume_data->_dim);
                sta.set_target_labels(std::vector<unsigned char>(1, label));
                sta.get_intensity_analysis(ellipsoid, pixel_num , min , max , mean , var , std);

                break;
            }
        default:
            {
                QTWIDGETS_THROW_EXCEPTION("Unsupported data type!");
            }
        }

        model->modify_intensity_info(i ,IntensityInfo(pixel_num , min , max , mean , var , std));
    }
}

MED_IMG_END_NAMESPACE