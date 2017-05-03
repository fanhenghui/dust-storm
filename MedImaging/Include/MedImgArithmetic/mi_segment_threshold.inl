
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
    T tmp;

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
                    tmp = _data_ref[idx];
                    if (tmp > threshold)
                    {
                        _mask_ref[idx] = _target_label;
                    }
                }
            }
        }
    }
}
