
template<class T>
void VolumeStatistician<T>::get_intensity_analysis(const unsigned int (&dim)[3] , T* data_array , const Sphere& sphere, 
    unsigned int& num_out , double& min_out , double& max_out , double& mean_out , double& var_out, double& std_out)
{
    const Point3& center = sphere._center;
    const double& radius = sphere._radius;

    unsigned int begin[3] , end[3];
    ArithmeticUtils::get_valid_region(dim , sphere , begin , end);

    T min0 = (std::numeric_limits<T>::max)();
    T max0 = (std::numeric_limits<T>::min)();
    double sum = 0.0;
    double num = 0.0;
    T tmp = 0;
    const unsigned int layer = dim[0]*dim[1];

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
                if (!((Point3(x,y,z) - center).magnitude() > radius))
                {
                    tmp = data_array[z*layer + y*dim[0] + x];
                    min0 = tmp < min0 ? tmp : min0;
                    max0 = tmp > max0 ? tmp : max0;
                    sum += static_cast<double>(tmp);
                    num += 1.0;
                }
            }
        }
    }
    num_out = static_cast<unsigned int>(num);
    mean_out = sum / num;
    min_out = static_cast<double>(min0);
    max_out = static_cast<double>(max0);

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
                if (!((Point3(x,y,z) - center).magnitude() > radius))
                {
                    tmp = data_array[z*layer + y*dim[0] + x];
                    sum += (tmp - mean_out)*(tmp - mean_out);
                }
            }
        }
    }
    var_out = sum / num;
    std_out = sqrt(var_out);
}

template<class T>
void VolumeStatistician<T>::get_intensity_analysis(const unsigned int (&dim)[3] , T* data_array , const Ellipsoid& ellipsoid, 
    unsigned int& num_out , double& min_out , double& max_out , double& mean_out , double& var_out, double& std_out)
{
    const Point3& center = ellipsoid._center;
    const double& aa_r = 1.0 / (ellipsoid._a*ellipsoid._a);
    const double& bb_r = 1.0 / (ellipsoid._b*ellipsoid._b);
    const double& cc_r = 1.0 / (ellipsoid._c*ellipsoid._c);

    unsigned int begin[3] , end[3];
    ArithmeticUtils::get_valid_region(dim , ellipsoid , begin , end);

    T min0 = (std::numeric_limits<T>::max)();
    T max0 = (std::numeric_limits<T>::min)();
    double sum = 0.0;
    double num = 0.0;
    T tmp = 0;
    Vector3 pt;
    const unsigned int layer = dim[0]*dim[1];

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
                pt = Point3(x,y,z) - center;
                if ( !(pt.x*pt.x * aa_r + pt.y*pt.y * bb_r + pt.z*pt.z * cc_r > 1.0) )
                {
                    tmp = data_array[z*layer + y*dim[0] + x];
                    min0 = tmp < min0 ? tmp : min0;
                    max0 = tmp > max0 ? tmp : max0;
                    sum += static_cast<double>(tmp);
                    num += 1.0;
                }
            }
        }
    }
    num_out = static_cast<unsigned int>(num);
    mean_out = sum / num;
    min_out = static_cast<double>(min0);
    max_out = static_cast<double>(max0);


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
                pt = Point3(x,y,z) - center;
                if ( !(pt.x*pt.x * aa_r + pt.y*pt.y * bb_r + pt.z*pt.z * cc_r > 1.0) )
                {
                    tmp = data_array[z*layer + y*dim[0] + x];
                    sum += (tmp - mean_out)*(tmp - mean_out);
                }
            }
        }
    }
    var_out = sum / num;
    std_out = sqrt(var_out);
}


