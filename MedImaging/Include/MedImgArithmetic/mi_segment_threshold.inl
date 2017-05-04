
template<class T>
void SegmentThreshold<T>::segment(const Ellipsoid& ellipsoid , T threshold)
{
    const Point3& center = ellipsoid._center;
    const double& aa_r = 1.0 / (ellipsoid._a*ellipsoid._a);
    const double& bb_r = 1.0 / (ellipsoid._b*ellipsoid._b);
    const double& cc_r = 1.0 / (ellipsoid._c*ellipsoid._c);

    unsigned int begin[3] , end[3];
    ArithmeticUtils::get_valid_region(_dim , ellipsoid , begin , end);

    const unsigned int layer = _dim[0]*_dim[1];

#ifndef _DEBUG
#pragma omp parallel for
#endif
    for (unsigned int z = begin[2] ; z < end[2] ; ++z)
    {
#ifndef _DEBUG
#pragma omp parallel for
#endif
        for (unsigned int y = begin[1] ; y < end[1] ; ++y)
        {
#ifndef _DEBUG
#pragma omp parallel for
#endif
            for (unsigned int x = begin[0] ; x < end[0] ; ++x)
            {
                Vector3 pt = Point3(x,y,z) - center;
                if ( !(pt.x*pt.x * aa_r + pt.y*pt.y * bb_r + pt.z*pt.z * cc_r > 1.0) )
                {
                    unsigned int idx  = z*layer + y*_dim[0] + x;
                    T tmp = _data_ref[idx];
                    if (tmp > threshold)
                    {
                        _mask_ref[idx] = _target_label;
                    }
                }
            }
        }
    }
}

template<class T>
void SegmentThreshold<T>::segment_auto_threshold(const Ellipsoid& ellipsoid)
{
    T threshold = get_auto_threshold_i(ellipsoid);
    segment(ellipsoid , threshold);
}

template<class T>
T SegmentThreshold<T>::get_auto_threshold_i(const Ellipsoid& ellipsoid)
{
    //double bound = (std::min)((std::min)(ellipsoid._a , ellipsoid._b) , ellipsoid._c);
    //bound = bound > 4.0 ? 4.0 : bound;
    //if (bound < 1.0)
    //{
    //    
    //}
    //else
    //{
    //    
    //}

    Sampler<T> sampler;
    float v = sampler.sample_3d_linear(ellipsoid._center.x , ellipsoid._center.y , ellipsoid._center.z , _dim[0] , _dim[1] , _dim[2] , _data_ref);
    v -= 200;
    return static_cast<T>(v);
}
