#include "mi_brick_info_generator.h"

#include "boost/thread.hpp"

#include "MedImgCommon/mi_concurrency.h"
#include "MedImgIO/mi_image_data.h"

#include "mi_brick_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

CPUVolumeBrickInfoGenerator::CPUVolumeBrickInfoGenerator()
{

}

CPUVolumeBrickInfoGenerator::~CPUVolumeBrickInfoGenerator()
{

}

void CPUVolumeBrickInfoGenerator::calculate_brick_info( 
std::shared_ptr<ImageData> image_data , 
unsigned int brick_size , 
unsigned int brick_expand , 
BrickCorner* _brick_corner_array , 
BrickUnit* brick_unit_array , 
VolumeBrickInfo* brick_info_array )
{
    RENDERALGO_CHECK_NULL_EXCEPTION(brick_unit_array);

    unsigned int brick_dim[3] = {1,1,1};
    BrickUtils::instance()->get_brick_dim(image_data->_dim , brick_dim , brick_size);

    const unsigned int brick_count = brick_dim[0]*brick_dim[1]*brick_dim[2];

    const unsigned int dispatch = Concurrency::instance()->get_app_concurrency();

    std::vector<boost::thread> threads(dispatch-1);
    const unsigned int brick_dispatch = brick_count/dispatch;

    switch(image_data->_data_type)
    {
    case UCHAR:
        {
            for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
            {
                threads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<unsigned char>, this , 
                    brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array , brick_info_array , image_data , brick_size , brick_expand));
            }
            calculate_brick_info_kernel_i<unsigned char>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array ,  brick_info_array , image_data , brick_size , brick_expand);
            std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    case USHORT:
        {
            for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
            {
                threads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<unsigned short>, this , 
                    brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array ,  brick_info_array , image_data , brick_size , brick_expand));
            }
            calculate_brick_info_kernel_i<unsigned short>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array ,  brick_info_array , image_data , brick_size , brick_expand);
            std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    case SHORT:
        {
            for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
            {
                threads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<short>, this , 
                    brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array ,  brick_info_array , image_data , brick_size , brick_expand));
            }
            calculate_brick_info_kernel_i<short>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array ,  brick_info_array , image_data , brick_size , brick_expand);
            std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    case FLOAT:
        {
            for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
            {
                threads[i] = boost::thread(boost::bind(&CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i<float>, this , 
                    brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array ,  brick_info_array , image_data , brick_size , brick_expand));
            }
            calculate_brick_info_kernel_i<float>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array ,  brick_info_array , image_data , brick_size , brick_expand);
            std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));

            break;
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Undefined data type!");
        }
    }
}


template<typename T>
void CPUVolumeBrickInfoGenerator::calculate_brick_info_i( 
    BrickCorner& bc , 
    BrickUnit& bu , 
    VolumeBrickInfo& vbi,
    std::shared_ptr<ImageData> image_data , 
    unsigned int brick_size , 
    unsigned int brick_expand )
{
    unsigned int *volume_dim = image_data->_dim;

    const unsigned int brick_length = brick_size + 2*brick_expand;

    unsigned int begin[3] = {bc.min[0] , bc.min[1], bc.min[2]};
    unsigned int end[3] = {bc.min[0]  + brick_size + brick_expand, bc.min[1] + brick_size + brick_expand, bc.min[2] + brick_size + brick_expand};

    unsigned int brick_begin[3] = {0,0,0};
    unsigned int brick_end[3] = {brick_length,brick_length,brick_length};

    for ( int i= 0 ; i < 3 ; ++i)
    {
        if (begin[i] < brick_expand)
        {
            brick_begin[i] += brick_expand - begin[i];
            begin[i] = 0;
        }
        else
        {
            begin[i] -= brick_expand;
        }

        if (end[i] > volume_dim[i])
        {
            brick_end[i] -= (end[i] - volume_dim[i]);
            end[i] = volume_dim[i];
        }
    }

    float max_scalar = -65535.0f;
    float min_scalar = 65535.0f;
    T* brick_data_array = (T*)bu.data;
    T cur_value = 0;
    float cur_value_float = 0;
    const unsigned int brick_layer_count = brick_length*brick_length;
    for (unsigned int z = brick_begin[2] ; z < brick_end[2] ; ++z)
    {
        for (unsigned int y = brick_begin[1] ; y < brick_end[1] ; ++y)
        {
            for (unsigned int x = brick_begin[0] ; x < brick_end[0] ; ++x)
            {
                cur_value = brick_data_array[z*brick_layer_count + y*brick_length + x];
                cur_value_float = (float)cur_value;
                max_scalar = cur_value_float > max_scalar ? cur_value_float : max_scalar;
                min_scalar = cur_value_float < min_scalar ? cur_value_float : min_scalar;
            }
        }
    }

    vbi.min = min_scalar;
    vbi.max = max_scalar;
}


template<typename T>
void CPUVolumeBrickInfoGenerator::calculate_brick_info_kernel_i( 
    unsigned int begin , 
    unsigned int end , 
    BrickCorner* _brick_corner_array , 
    BrickUnit* brick_unit_array , 
    VolumeBrickInfo* brick_info_array,
    std::shared_ptr<ImageData> image_data , 
    unsigned int brick_size , 
    unsigned int brick_expand )
{
    for (unsigned int i = begin ; i < end ; ++i)
    {
        calculate_brick_info_i<T>(_brick_corner_array[i] , brick_unit_array[i] ,brick_info_array[i] , image_data , brick_size , brick_expand);
    }
}



MED_IMAGING_END_NAMESPACE