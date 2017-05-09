#include "MedImgArithmetic/mi_volume_statistician.h"
#include <limits>

using namespace MED_IMAGING_NAMESPACE;

void UT_Statistic()
{
    unsigned int x = 512;
    unsigned int y = 512;
    unsigned int z = 734;
    std::ifstream in("D:/temp/AB_CTA_01.raw", std::ios::binary | std::ios::out);
    if (!in.is_open())
    {
        std::cout << "Open file failed!\n";
        return;
    }
    unsigned short* data_array = new unsigned short[x*y*z];
    in.read((char*)data_array, x*y*z*2);
    in.close();

    VolumeStatistician<unsigned short> vs;
    double min , max , mean , var , std;
    unsigned int num;
    const unsigned int dim[3] = {512,512,734};
    vs.set_data_ref(data_array);
    vs.set_mask_ref(nullptr);
    vs.set_dim(dim);

    Ellipsoid ellipsoid;
    ellipsoid._center = Point3(256.0,256.0 ,256.0);
    ellipsoid._a = 80;
    ellipsoid._b = 20;
    ellipsoid._c = 50;

    vs.get_intensity_analysis(ellipsoid , num , min , max , mean , var , std);

    std::cout << "num : " << num << std::endl;
    std::cout << "min : " << min << std::endl;
    std::cout << "max : " << max << std::endl;
    std::cout << "mean : " << mean<< std::endl;
    std::cout << "variance : " << var << std::endl;
    std::cout << "standard variance : " << std << std::endl << std::endl;



}