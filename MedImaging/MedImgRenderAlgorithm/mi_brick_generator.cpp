#include "mi_brick_generator.h"
#include "mi_brick_utils.h"
#include "MedImgIO/mi_image_data.h"

#include "boost/thread.hpp"
#include "MedImgCommon/mi_concurrency.h"

MED_IMAGING_BEGIN_NAMESPACE

BrickGenerator::BrickGenerator()
{

}

BrickGenerator::~BrickGenerator()
{

}

 void BrickGenerator::calculate_brick_corner( 
     std::shared_ptr<ImageData> image_data , 
     unsigned int brick_size , unsigned int brick_expand , 
     BrickCorner* _brick_corner_array)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(_brick_corner_array);

    unsigned int brick_dim[3] = {1,1,1};
    BrickUtils::instance()->get_brick_dim(image_data->_dim , brick_dim , brick_size);

    const unsigned int brick_count = brick_dim[0]*brick_dim[1]*brick_dim[2];

    //BrickCorner* _brick_corner_array = new BrickCorner[brick_count];
    unsigned int x , y , z;
    unsigned int layer_count = brick_dim[0]*brick_dim[1];
    for (unsigned int i = 0 ; i<brick_count ; ++i)
    {
        z =  i/layer_count;
        y = (i - z*layer_count)/brick_dim[0];
        x = i - z*layer_count - y*brick_dim[0];
        _brick_corner_array[i].min[0] = x*brick_size;
        _brick_corner_array[i].min[1] = y*brick_size;
        _brick_corner_array[i].min[2] = z*brick_size;
    }

}

void BrickGenerator::calculate_brick_unit( 
    std::shared_ptr<ImageData> image_data ,
    BrickCorner* _brick_corner_array , 
    unsigned int brick_size , unsigned int brick_expand , 
    BrickUnit* brick_unit_array)
{
    RENDERALGO_CHECK_NULL_EXCEPTION(brick_unit_array);

    unsigned int brick_dim[3] = {1,1,1};
    BrickUtils::instance()->get_brick_dim(image_data->_dim , brick_dim , brick_size);

    const unsigned int brick_count = brick_dim[0]*brick_dim[1]*brick_dim[2];
    //BrickUnit* brick_unit_array = new BrickUnit[brick_count];

    const unsigned int dispatch = Concurrency::instance()->get_app_concurrency();

    std::vector<boost::thread> threads(dispatch-1);
    const unsigned int brick_dispatch = brick_count/dispatch;

    switch(image_data->_data_type)
    {
    case UCHAR:
        {
            if (1 == dispatch)
            {
                calculate_brick_unit_kernel_i<unsigned char>(0 , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
                {
                    threads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<unsigned char>, this , 
                        brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand));
                }
                calculate_brick_unit_kernel_i<unsigned char>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
                std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));
            }

            break;
        }
    case USHORT:
        {
            if (1 == dispatch)
            {
                calculate_brick_unit_kernel_i<unsigned short>(0 , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
                {
                    threads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<unsigned short>, this , 
                        brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand));
                }
                calculate_brick_unit_kernel_i<unsigned short>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
                std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));
            }
            break;
        }
    case SHORT:
        {
            if (1 == dispatch)
            {
                calculate_brick_unit_kernel_i<short>(0 , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
                {
                    threads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<short>, this , 
                        brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand));
                }
                calculate_brick_unit_kernel_i<short>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
                std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));
            }
            break;
        }
    case FLOAT:
        {
            if (1 == dispatch)
            {
                calculate_brick_unit_kernel_i<float>(0 , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
            }
            else
            {
                for ( unsigned int i = 0 ; i < dispatch - 1 ; ++i)
                {
                    threads[i] = boost::thread(boost::bind(&BrickGenerator::calculate_brick_unit_kernel_i<float>, this , 
                        brick_dispatch*i , brick_dispatch*(i+1) , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand));
                }
                calculate_brick_unit_kernel_i<float>(brick_dispatch*(dispatch-1) , brick_count , _brick_corner_array , brick_unit_array , image_data , brick_size , brick_expand);
                std::for_each(threads.begin() , threads.end() , std::mem_fun_ref(&boost::thread::join));
            }
            break;
        }
    default:
        {
            RENDERALGO_THROW_EXCEPTION("Undefined data type!");
        }
    }
}

template<typename T>
void BrickGenerator::calculate_brick_unit_kernel_i( 
    unsigned int begin , unsigned int end ,
    BrickCorner* _brick_corner_array , BrickUnit* brick_unit_array , 
    std::shared_ptr<ImageData> image_data , 
    unsigned int brick_size , unsigned int brick_expand )
{
    for (unsigned int i = begin ; i < end ; ++i)
    {
        calculate_brick_unit_i<T>(_brick_corner_array[i] , brick_unit_array[i] ,image_data , brick_size , brick_expand);
    }
}

template<typename T>
void BrickGenerator::calculate_brick_unit_i( 
    BrickCorner& bc , BrickUnit& bu , 
    std::shared_ptr<ImageData> image_data , 
    unsigned int brick_size , unsigned int brick_expand )
{
    const unsigned int brick_length = brick_size + 2*brick_expand;
    bu.data = new char[sizeof(T)*brick_length*brick_length*brick_length];
    memset(bu.data , 0 , sizeof(T)*(brick_length*brick_length*brick_length));

    T* pDst = (T*)bu.data;
    T* pSrc = (T*)image_data->get_pixel_pointer();
    unsigned int *volume_dim = image_data->_dim;

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

    assert(end[0] - begin[0] == brick_end[0] - brick_begin[0]);
    assert(end[1] - begin[1] == brick_end[1] - brick_begin[1]);
    assert(end[2] - begin[2] == brick_end[2] - brick_begin[2]);

    const unsigned int volume_layer_count = volume_dim[0]*volume_dim[1];
    const unsigned int brick_layer_count = brick_length*brick_length;
    const unsigned int copy_length = brick_end[0] - brick_begin[0];
    const unsigned int copy_bytes = copy_length*sizeof(T);

    for (unsigned int z = begin[2] , z0 = brick_begin[2] ; z < end[2] ; ++z , ++z0)
    {
        for (unsigned int y = begin[1] , y0 = brick_begin[1] ; y < end[1] ; ++y , ++y0)
        {
            memcpy(pDst+z0*brick_layer_count + y0*brick_length + brick_begin[0] , pSrc + z*volume_layer_count + y*volume_dim[0] + begin[0] , copy_bytes);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //Test output
    //{
    //    std::stringstream ss;
    //    ss << "D:/temp/brick_data_" << bc.m_Min[0] <<"_"<< bc.m_Min[1] <<"_"<< bc.m_Min[2] <<".raw";
    //    std::ofstream out(ss.str().c_str() , std::ios::out | std::ios::binary);
    //    if (out.is_open())
    //    {
    //        out.write((char*)pDst , sizeof(T)*brick_length*brick_length*brick_length);
    //        out.close();
    //    }
    //}
    //////////////////////////////////////////////////////////////////////////
}



MED_IMAGING_END_NAMESPACE