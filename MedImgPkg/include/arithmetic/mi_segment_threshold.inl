
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
                if ( pt.x*pt.x * aa_r + pt.y*pt.y * bb_r + pt.z*pt.z * cc_r < 1.0 )
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
void SegmentThreshold<T>::segment_auto_threshold(const Ellipsoid& ellipsoid , ThresholdType type)
{
    T threshold = 0;
    switch(type)
    {
    case Center:
        {
            threshold = get_threshold_center_i(ellipsoid);
            break;
        }
    case Otsu:
        {
            threshold = get_threshold_otsu_i(ellipsoid);
            std::cout << "Auto Otsu threshold is : " << threshold <<std::endl;
            break;
        }
    default:
        break;
    }
    segment(ellipsoid , threshold);
}

template<class T>
T SegmentThreshold<T>::get_threshold_center_i(const Ellipsoid& ellipsoid)
{
    Sampler<T> sampler;
    float v = sampler.sample_3d_linear(ellipsoid._center.x , ellipsoid._center.y , ellipsoid._center.z , _dim[0] , _dim[1] , _dim[2] , _data_ref);
    v -= 200;
    return static_cast<T>(v);
}

template<class T>
T SegmentThreshold<T>::get_threshold_otsu_i(const Ellipsoid& ellipsoid)
{
    const Point3& center = ellipsoid._center;
    const double& aa_r = 1.0 / (ellipsoid._a*ellipsoid._a);
    const double& bb_r = 1.0 / (ellipsoid._b*ellipsoid._b);
    const double& cc_r = 1.0 / (ellipsoid._c*ellipsoid._c);

    unsigned int begin[3] , end[3];
    ArithmeticUtils::get_valid_region(_dim , ellipsoid , begin , end);
    
    const unsigned int layer = _dim[0]*_dim[1];

    //1 Calculate histogram
    const int gray_level = 256;
    //boost::atomic_uint32_t gray_hist[gray_level];
    unsigned int gray_hist[gray_level];
    memset(gray_hist , 0 , sizeof(gray_hist));
    int pixel_num = 0;
    float sum = 0;

    const float ww = _max_scalar - _min_scalar;
    const float ww_r = 1.0f/ww;

//#ifndef _DEBUG
//#pragma omp parallel for
//#endif
    for (unsigned int z = begin[2] ; z < end[2] ; ++z)
    {
//#ifndef _DEBUG
//#pragma omp parallel for
//#endif
        for (unsigned int y = begin[1] ; y < end[1] ; ++y)
        {
//#ifndef _DEBUG
//#pragma omp parallel for
//#endif
            for (unsigned int x = begin[0] ; x < end[0] ; ++x)
            {
                Vector3 pt = Point3(x,y,z) - center;
                if ( !(pt.x*pt.x * aa_r + pt.y*pt.y * bb_r + pt.z*pt.z * cc_r > 1.0) )
                {
                    unsigned int idx  = z*layer + y*_dim[0] + x;
                    T tmp = _data_ref[idx];
                    float temp_norm = (tmp - _min_scalar)*ww_r;
                    temp_norm = temp_norm > 1.0f ? 1.0f : temp_norm;
                    temp_norm = temp_norm < 0.0f ? 0.0f: temp_norm;
                    int gray_scalar = int(temp_norm*(gray_level-1));
                    gray_hist[gray_scalar] = gray_hist[gray_scalar]+1;
                    sum += gray_scalar * 0.001f;
                    ++pixel_num;
                }
            }
        }
    }

    //2 Loop to find ICA 
    // Nobuyuki Otsu define : ICA = PA*(MA-M)^2 + PB*(MB-M)^2
    // Part A less than threshold t , and part B large than threshold t
    // M : all pixel mean 
    // MA : part A mean
    // MB : part B mean
    // PA =  part A pixel / all pixel
    // PB = part B pixel / all pixel

    //calculate mean
    const float pixel_num_f = (float)pixel_num;
    const float mean = sum / (float)pixel_num * 1000.0f;
    float max_icv = std::numeric_limits<float>::min();
    int max_gray_scalar = -1;
    for (int i = 1 ; i< gray_level-1 ; ++i)
    {
        //PA MA
        unsigned int pixel_a = 0;
        float sum_a = 0.0f;
        for (int j = 0 ; j <= i ; ++j)
        {
            pixel_a += gray_hist[j];
            sum_a += (float)j*gray_hist[j]*0.001f;
        }
        float pa = (float)pixel_a / pixel_num_f;
        float mean_a = sum_a / (float)pixel_a * 1000.0f;

        //PB MB
        float pixel_b_f = (float)(pixel_num - pixel_a);
        float pb = pixel_b_f / pixel_num_f;
        float mean_b = (sum - sum_a) / pixel_b_f * 1000.0f;

        //ICV
        float icv = pa * (mean_a - mean)*(mean_a - mean) + 
            pb * (mean_b - mean)*(mean_b - mean);

        if (icv > max_icv)
        {
            max_icv = icv;
            max_gray_scalar = i;
        }
    }

    float gray_norm = (float)max_gray_scalar / (float)gray_level;
    return T(gray_norm * ww + _min_scalar);
}
