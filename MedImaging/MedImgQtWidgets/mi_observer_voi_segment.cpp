#include "mi_observer_voi_segment.h"

#include "MedImgCommon/mi_configuration.h"

#include "MedImgArithmetic/mi_segment_threshold.h"

#include "MedImgIO/mi_image_data.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"


#include "mi_model_voi.h"

MED_IMAGING_BEGIN_NAMESPACE

VOISegmentObserver::VOISegmentObserver()
{

}

VOISegmentObserver::~VOISegmentObserver()
{

}

void VOISegmentObserver::set_model(std::shared_ptr<VOIModel> model)
{
    _model = model;
}

void VOISegmentObserver::set_volume_infos(std::shared_ptr<VolumeInfos> volume_infos)
{
    _volume_infos = volume_infos;
}

void VOISegmentObserver::update(int code_id /*= 0*/)
{
    //VOIModel::print_code_id(code_id);
    try
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_volume_infos);

        std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
        QTWIDGETS_CHECK_NULL_EXCEPTION(volume_data);

        std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();
        QTWIDGETS_CHECK_NULL_EXCEPTION(camera_cal);

        std::shared_ptr<VOIModel> model = _model.lock();
        QTWIDGETS_CHECK_NULL_EXCEPTION(model);

        const std::vector<VOISphere>& vois = model->get_vois();
        const std::vector<unsigned char>& labels = model->get_labels();

        //1 Update overlay mask mode
        if (VOIModel::ADD_VOI == code_id || VOIModel::DELETE_VOI == code_id)
        {
            if (vois.empty())
            {
                for (auto it = _scenes.begin() ; it != _scenes.end() ; ++it)
                {
                    (*it)->set_mask_overlay_mode(MASK_OVERLAY_DISABLE);
                }
            }
            else
            {
                for (auto it = _scenes.begin() ; it != _scenes.end() ; ++it)
                {
                    (*it)->set_mask_overlay_mode(MASK_OVERLAY_ENABLE);
                }
            }
        }
        //Add VOI
        if (VOIModel::ADD_VOI == code_id)
        {
            //get added VOI form list rear
            const VOISphere voi_added = model->get_voi(vois.size() - 1);
            const unsigned char label_added = model->get_label(vois.size() - 1);

            if (voi_added.diameter < 0.1f)
            {
                _pre_voi_aabbs[label_added] = AABBUI();
                _pre_vois = vois;
            }
            else
            {
                assert(_pre_voi_aabbs.find(label_added) == _pre_voi_aabbs.end());

                Ellipsoid ellipsoid = voi_patient_to_volume(voi_added);
                AABBUI aabb = get_aabb_i(ellipsoid);

                segment_i(ellipsoid , aabb , label_added);

                _pre_voi_aabbs[label_added] = aabb;
                _pre_vois = vois;
            }

            //Update visible labels
            for (auto it = _scenes.begin() ; it != _scenes.end() ; ++it)
            {
                (*it)->set_mask_overlay_color(RGBAUnit(1.0f,0.0f,0.0f) , label_added);
                if (model->is_voi_mask_visible())
                {
                    (*it)->set_visible_labels(labels);
                }
                else
                {
                    (*it)->set_visible_labels(std::vector<unsigned char>());
                }
            }

        }


        //Delete VOI
        if (VOIModel::DELETE_VOI == code_id)
        {
            if (_pre_vois.empty())
            {
                return;
            }

            //get deleted VOI from compare
            auto it_deleted = _pre_voi_aabbs.begin();
            for (; it_deleted != _pre_voi_aabbs.end() ; ++it_deleted)
            {
                unsigned char temp_label = it_deleted->first;
                bool constant = false;
                for (auto it2 = labels.begin() ; it2 != labels.end() ; ++it2)
                {
                    if (temp_label == (*it2))
                    {
                        constant = true;
                        break;
                    }
                }

                if (!constant)
                {
                    break;
                }
            }

            unsigned char label_deleted = it_deleted->first;
            AABBUI aabb_deleted = it_deleted->second;

            assert(label_deleted != 0);

            //Recover mask
            recover_i(aabb_deleted , label_deleted);
            _pre_voi_aabbs.erase(it_deleted);
            _pre_vois = vois;

            //Update visible labels
            for (auto it = _scenes.begin() ; it != _scenes.end() ; ++it)
            {
                if (model->is_voi_mask_visible())
                {
                    (*it)->set_visible_labels(labels);
                }
                else
                {
                    (*it)->set_visible_labels(std::vector<unsigned char>());
                }
            }
        }

        //Modifying
        if (VOIModel::MODIFYING == code_id)
        {
            //do nothing
        }

        //Modify completed
        if (VOIModel::MODIFY_COMPLETED == code_id)
        {
            assert(_pre_vois.size() == vois.size());

            int idx = -1;
            for (int i = 0 ; i < vois.size() ; ++i)
            {
                if (_pre_vois[i] != vois[i])
                {
                    idx = i;
                    break;
                }
            }

            //Modify none
            if (idx == -1)
            {
                return;
            }

            const unsigned char label_modify = model->get_label(idx);

            //1 Recover
            recover_i(_pre_voi_aabbs[label_modify] , label_modify);

            //2 Segment
            Ellipsoid ellipsoid = voi_patient_to_volume(model->get_voi(idx));
            AABBUI aabb = get_aabb_i(ellipsoid);

            segment_i(ellipsoid , aabb , label_modify);

            _pre_voi_aabbs[label_modify] = aabb;
            _pre_vois = vois;

            for (auto it = _scenes.begin() ; it != _scenes.end() ; ++it)
            {
                (*it)->set_dirty(true);
            }
        }

        if (VOIModel::TUNING_VOI == code_id)
        {
            // _pre_vois (aka sphere geometry) do not change at all
            assert(_pre_vois.size() == vois.size());// actually not only size, each element of _pre_vois is exactly the same as that of vois
            AABBUI voxel_block = model->get_voxel_to_tune();// copy one
            int voi_idx = model->get_voi_to_tune();
            
            if (voi_idx < 0)
            {
                return;
            }
            const unsigned char voi_label = model->get_label(voi_idx);
            const AABBUI& voi_boundingbox = _pre_voi_aabbs[voi_label]; 
            //intersect voi_boundingbox with voxel_block
            int intersect = voxel_block.Intersect(voi_boundingbox);
            if (intersect)
            {
                recover_i( voxel_block, voi_label);

                for (auto it = _scenes.begin() ; it != _scenes.end() ; ++it)
                {
                    (*it)->set_dirty(true);
                }
            }
        }
    }
    catch (const Exception& e)
    {
        //TODO LOG
        std::cout << "VOI segment OB update failed! " << e.what();
        assert(false);
        throw e;
    }
}

void VOISegmentObserver::set_scenes(std::vector<std::shared_ptr<MPRScene>> scenes)
{
    _scenes = scenes;
}

Ellipsoid VOISegmentObserver::voi_patient_to_volume(const VOISphere& voi)
{
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    std::shared_ptr<CameraCalculator> camera_cal = _volume_infos->get_camera_calculator();

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

AABBUI VOISegmentObserver::get_aabb_i(const Ellipsoid& ellipsoid)
{
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();

    unsigned int begin[3] , end[3];
    ArithmeticUtils::get_valid_region(volume_data->_dim , ellipsoid , begin , end);

    return AABBUI(begin , end);
}
void VOISegmentObserver::recover_i(const AABBUI& aabb , unsigned char label)
{
    std::shared_ptr<ImageData> mask_data = _volume_infos->get_mask();
    unsigned char* mask_array = (unsigned char*)mask_data->get_pixel_pointer();
    const unsigned int layer = mask_data->_dim[0]*mask_data->_dim[1];

#ifndef _DEBUG
#pragma omp parallel for
#endif
    for (unsigned int z = aabb._min[2] ; z < aabb._max[2] ; ++z)
    {
#ifndef _DEBUG
#pragma omp parallel for
#endif
        for (unsigned int y = aabb._min[1] ; y < aabb._max[1] ; ++y)
        {
#ifndef _DEBUG
#pragma omp parallel for
#endif
            for (unsigned int x = aabb._min[0] ; x < aabb._max[0] ; ++x)
            {
                unsigned int idx = z*layer + y*mask_data->_dim[0] + x;
                if (mask_array[idx] == label)
                {
                    mask_array[idx] = 0;
                }
            }
        }
    }

    //Update to texture
    if (GPU == Configuration::instance()->get_processing_unit_type())
    {
        if (aabb != AABBUI())
        {
            update_aabb_i(aabb);
        }
    }
}

void VOISegmentObserver::segment_i(const Ellipsoid& ellipsoid , const AABBUI& aabb ,unsigned char label)
{
    std::shared_ptr<ImageData> volume_data = _volume_infos->get_volume();
    std::shared_ptr<ImageData> mask_data = _volume_infos->get_mask();
    const DataType data_type = volume_data->_data_type;

    switch(data_type)
    {
    case SHORT:
        {
            //get threshold
            SegmentThreshold<short> segment;
            segment.set_data_ref((short*)volume_data->get_pixel_pointer());
            segment.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
            segment.set_dim(volume_data->_dim);
            segment.set_target_label(label);
            segment.set_min_scalar(volume_data->get_min_scalar());
            segment.set_max_scalar(volume_data->get_max_scalar());
            segment.segment_auto_threshold(ellipsoid , SegmentThreshold<short>::Otsu);
            break;
        }
    case USHORT:
        {
            //get threshold
            SegmentThreshold<unsigned short> segment;
            segment.set_data_ref((unsigned short*)volume_data->get_pixel_pointer());
            segment.set_mask_ref((unsigned char*)mask_data->get_pixel_pointer());
            segment.set_dim(volume_data->_dim);
            segment.set_target_label(label);
            segment.set_min_scalar(volume_data->get_min_scalar());
            segment.set_max_scalar(volume_data->get_max_scalar());
            segment.segment_auto_threshold(ellipsoid, SegmentThreshold<unsigned short>::Otsu);

            break;
        }
    default:
        QTWIDGETS_THROW_EXCEPTION("Unsupported data type!");
    }

    //Update to texture
    if (GPU == Configuration::instance()->get_processing_unit_type())
    {
        if (aabb != AABBUI())
        {
            update_aabb_i(aabb);
        }
    }
}

void VOISegmentObserver::update_aabb_i(const AABBUI& aabb)
{
    unsigned int dim_brick[3] = {aabb._max[0] - aabb._min[0],
        aabb._max[1] - aabb._min[1],
        aabb._max[2] - aabb._min[2]};
    unsigned char* mask_updated = new unsigned char[dim_brick[0]*dim_brick[1]*dim_brick[2]];

    std::shared_ptr<ImageData> mask_data = _volume_infos->get_mask();
    unsigned char* mask_array = (unsigned char*)mask_data->get_pixel_pointer();

    const unsigned int layer_whole = mask_data->_dim[0]*mask_data->_dim[1];
    const unsigned int layer_brick = dim_brick[0]*dim_brick[1];

    for(unsigned int z = aabb._min[2] ; z < aabb._max[2] ; ++z)
    {
        for(unsigned int y = aabb._min[1] ; y < aabb._max[1] ; ++y)
        {
            memcpy( mask_updated + (z-aabb._min[2])*layer_brick + (y - aabb._min[1])*dim_brick[0],
                mask_array + z*layer_whole + y*mask_data->_dim[0] + aabb._min[0] ,
                dim_brick[0]);
        }
    }

    _volume_infos->update_mask(aabb._min , aabb._max , mask_updated);

}


MED_IMAGING_END_NAMESPACE