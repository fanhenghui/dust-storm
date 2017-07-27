#include "mi_brick_pool.h"

#include "MedImgUtil/mi_configuration.h"

#include "MedImgIO/mi_image_data.h"

#include "mi_brick_info_generator.h"

MED_IMG_BEGIN_NAMESPACE

BrickPool::BrickPool():_brick_size(32),_brick_expand(4),_brick_count(0)
{
    _brick_dim[0] = 1;
    _brick_dim[1] = 1;
    _brick_dim[2] = 1;
}

BrickPool::~BrickPool()
{

}

void BrickPool::set_volume( std::shared_ptr<ImageData> image_data )
{
    _volume = image_data;
}

void BrickPool::set_mask( std::shared_ptr<ImageData> image_data )
{
    _mask = image_data;
}

void BrickPool::get_brick_dim(unsigned int (&brick_dim)[3])
{
    memcpy(brick_dim , _brick_dim , sizeof(unsigned int)*3);
}

unsigned int BrickPool::get_brick_count() const
{
    return _brick_count;
}

void BrickPool::set_brick_size( unsigned int brick_size )
{
    _brick_size= brick_size;
}

unsigned int BrickPool::get_brick_size() const
{
    return _brick_size;
}

void BrickPool::set_brick_expand( unsigned int brick_expand )
{
    _brick_expand = brick_expand;
}

unsigned int BrickPool::get_brick_expand() const
{
    return _brick_expand;
}

BrickCorner* BrickPool::get_brick_corner()
{
    return _brick_corner_array.get();
}

VolumeBrickInfo* BrickPool::get_volume_brick_info()
{
    return _volume_brick_info_array.get();
}

MaskBrickInfo* BrickPool::get_mask_brick_info( const std::vector<unsigned char>& vis_labels )
{
    LabelKey key(vis_labels);
    auto it = _mask_brick_info_array_set.find(key);
    if (it == _mask_brick_info_array_set.end())
    {
        return nullptr;
    }
    else
    {
        return it->second.get();
    }
}

void BrickPool::calculate_brick_corner()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume);

        for (int i = 0 ; i< 3 ; ++i)
        {
            _brick_dim[i] = (unsigned int)floor((float)_volume->_dim[i]/(float)_brick_size);
        }

        _brick_count = _brick_dim[0]*_brick_dim[1]*_brick_dim[2];
        _brick_corner_array.reset(new BrickCorner[_brick_count]);

        unsigned int x , y , z;
        const unsigned int layer_count = _brick_dim[0]*_brick_dim[1];
        for (unsigned int i = 0 ; i<_brick_count ; ++i)
        {
            z =  i/layer_count;
            y = (i - z*layer_count)/_brick_dim[0];
            x = i - z*layer_count - y*_brick_dim[0];
            _brick_corner_array[i].min[0] = x*_brick_size;
            _brick_corner_array[i].min[1] = y*_brick_size;
            _brick_corner_array[i].min[2] = z*_brick_size;
        }

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

void BrickPool::calculate_brick_geometry()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume);
        const unsigned int *dim = _volume->_dim;

        const unsigned int vertex_count = (_brick_dim[0] + 1) * (_brick_dim[1] + 1) * (_brick_dim[2] + 1);
        _brick_geometry.vertex_count = vertex_count;//vertex count is not the same with brick count

        _brick_geometry.vertex_array = new float[vertex_count*3];
        _brick_geometry.color_array = new float[vertex_count*4];
        _brick_geometry.brick_idx_units = new BrickEleIndex[_brick_count];

        
        float* vertex_array = _brick_geometry.vertex_array;
        float* color_array = _brick_geometry.color_array;
        BrickEleIndex *element_array = _brick_geometry.brick_idx_units;

        //vertex
        const float brick_size= static_cast<float>(_brick_size);
        float fx(0.0f),fy(0.0f),fz(0.0f);
        for (unsigned int z = 0 ; z < _brick_dim[2] + 1 ; ++z)
        {
            fz = ( z == _brick_dim[2] ) ? dim[2] : static_cast<float>(z)*brick_size;
            for (unsigned int y = 0 ; y < _brick_dim[1] + 1 ; ++y)
            {
                fy = ( y == _brick_dim[1] ) ?  dim[1] : static_cast<float>(y)*brick_size;
                for (unsigned int x = 0 ; x < _brick_dim[0] + 1 ; ++x)
                {
                    fx = ( x == _brick_dim[0] ) ? dim[0] : static_cast<float>(x)*brick_size;
                    const unsigned  int vertex_id = z*(_brick_dim[0]+1)*(_brick_dim[1]+1) + y*(_brick_dim[0]+1) + x;
                    vertex_array[vertex_id*3] = fx;
                    vertex_array[vertex_id*3+1] = fy;
                    vertex_array[vertex_id*3+2] = fz;
                }
            }
        }

        //set vertex coordinate as color (not nomalized)
        for (unsigned int i = 0; i < vertex_count ; ++i)
        {
            color_array[i*4] = vertex_array[i*3];
            color_array[i*4+1] = vertex_array[i*3+1];
            color_array[i*4+2] = vertex_array[i*3+2];
            color_array[i*4+3] = 1.0;
        }

        //element
#define VertexID(pt0,pt1,pt2) (pt2[2]*(_brick_dim[0]+1)*(_brick_dim[1]+1) + pt1[1]*(_brick_dim[0]+1) + pt0[0] )
        for (unsigned int z = 0 ; z < _brick_dim[2] ; ++z)
        {
            for (unsigned int y = 0 ; y < _brick_dim[1] ; ++y)
            {
                for (unsigned int x = 0 ; x < _brick_dim[0] ; ++x)
                {
                    const unsigned int idx = z*_brick_dim[0]*_brick_dim[1] + y*_brick_dim[0] + x;
                    const unsigned int ptmin[3] = {x , y , z};
                    const unsigned int ptmax[3] = {x+1 , y+1 , z+1};

                    unsigned int *pIdx = element_array[idx].idx;
                    pIdx[0] = VertexID(ptmax,ptmin,ptmax);
                    pIdx[1] = VertexID(ptmax,ptmin,ptmin);
                    pIdx[2] = VertexID(ptmax,ptmax,ptmin);
                    pIdx[3] = VertexID(ptmax,ptmin,ptmax);
                    pIdx[4] = VertexID(ptmax,ptmax,ptmin);
                    pIdx[5] = VertexID(ptmax,ptmax,ptmax);
                    pIdx[6] = VertexID(ptmax,ptmax,ptmax);
                    pIdx[7] = VertexID(ptmax,ptmax,ptmin);
                    pIdx[8] = VertexID(ptmin,ptmax,ptmin);
                    pIdx[9] = VertexID(ptmax,ptmax,ptmax);
                    pIdx[10] = VertexID(ptmin,ptmax,ptmin);
                    pIdx[11] = VertexID(ptmin,ptmax,ptmax);
                    pIdx[12] = VertexID(ptmax,ptmax,ptmax);
                    pIdx[13] = VertexID(ptmin,ptmax,ptmax);
                    pIdx[14] = VertexID(ptmin,ptmin,ptmax);
                    pIdx[15] = VertexID(ptmax,ptmax,ptmax);
                    pIdx[16] = VertexID(ptmin,ptmin,ptmax);
                    pIdx[17] = VertexID(ptmax,ptmin,ptmax);
                    pIdx[18] = VertexID(ptmin,ptmin,ptmax);
                    pIdx[19] = VertexID(ptmin,ptmax,ptmax);
                    pIdx[20] = VertexID(ptmin,ptmax,ptmin);
                    pIdx[21] = VertexID(ptmin,ptmin,ptmax);
                    pIdx[22] = VertexID(ptmin,ptmax,ptmin);
                    pIdx[23] = VertexID(ptmin,ptmin,ptmin);
                    pIdx[24] = VertexID(ptmin,ptmin,ptmax);
                    pIdx[25] = VertexID(ptmin,ptmin,ptmin);
                    pIdx[26] = VertexID(ptmax,ptmin,ptmin);
                    pIdx[27] = VertexID(ptmin,ptmin,ptmax);
                    pIdx[28] = VertexID(ptmax,ptmin,ptmin);
                    pIdx[29] = VertexID(ptmax,ptmin,ptmax);
                    pIdx[30] = VertexID(ptmin,ptmin,ptmin);
                    pIdx[31] = VertexID(ptmin,ptmax,ptmin);
                    pIdx[32] = VertexID(ptmax,ptmax,ptmin);
                    pIdx[33] = VertexID(ptmin,ptmin,ptmin);
                    pIdx[34] = VertexID(ptmax,ptmax,ptmin);
                    pIdx[35] = VertexID(ptmax,ptmin,ptmin);
                }
            }
        }
#undef VertexID

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

void BrickPool::calculate_volume_brick()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume);

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

void BrickPool::calculate_mask_brick()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(_volume);

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        assert(false);
        throw e;
    }
}

void BrickPool::update_mask_brick(unsigned int (&begin)[3] , unsigned int (&end)[3] , LabelKey label_key)
{

}

MED_IMG_END_NAMESPACE