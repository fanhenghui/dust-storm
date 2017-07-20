#include "mi_connected_domain_analysis.h"
#include <stack>

MED_IMAGING_BEGIN_NAMESPACE

void ConnectedDomainAnalysis::seed_filling(std::stack<POS>& s , unsigned char label , unsigned int& cd_num )
{
    cd_num = 0;
    while (!s.empty())
    {
        POS pos = s.top();
        s.pop();
        unsigned int add = 0;
        get_27_n(pos.x , pos.y , pos.z , s , label , add);
        if(s.empty() && cd_num == 0)
        {
            _roi_cache.get()[pos.z*_roi_dim[0]*_roi_dim[1] + pos.y*_roi_dim[0] +pos.x] = 0;//isolated point
            std::cout << "isolated point\n";
            break;
        }
        _roi_cache.get()[pos.z*_roi_dim[0]*_roi_dim[1] + pos.y*_roi_dim[0] +pos.x] = label;
        cd_num+= (add + 1);
    }

}

void ConnectedDomainAnalysis::keep_major()
{
    unsigned char* roi = _roi_cache.get();
    ARITHMETIC_CHECK_NULL_EXCEPTION(roi);

    //Seed filing
    std::map<unsigned char ,  unsigned int> tmp_cd;
    std::stack<POS> cur_cd;
    unsigned int cur_num = 0;
    unsigned char cur_label = 2;
    for (unsigned int z = 0 ; z< _roi_dim[2]; ++z)
    {
        for (unsigned int y = 0 ; y < _roi_dim[1] ; ++y)
        {
            for (unsigned int x = 0 ; x < _roi_dim[0] ; ++x)
            {
                unsigned int pos = z*_roi_dim[0]*_roi_dim[1] + y*_roi_dim[0] +x;
                unsigned char target_label = roi[pos];
                if(target_label == 1)
                {
                    cur_cd.push(POS(x,y,z));
                    seed_filling(cur_cd , cur_label , cur_num);
                    if (cur_num > 0)
                    {
                        tmp_cd[cur_label] = cur_num;
                        cur_num = 0;
                        cur_label++;
                    }
                }
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //Test
    //{
    //    int err_0 = 0;
    //    int err_1 = 0;
    //    int err_2 = 0;

    //    for (unsigned int z = 0 ; z< _roi_dim[2]; ++z)
    //    {
    //        for (unsigned int y = 0 ; y < _roi_dim[1] ; ++y)
    //        {
    //            for (unsigned int x = 0 ; x < _roi_dim[0] ; ++x)
    //            {
    //                unsigned int zz = z + _min[2];
    //                unsigned int yy = y + _min[1];
    //                unsigned int xx = x + _min[0];
    //                unsigned int idx0 = z*_roi_dim[0]*_roi_dim[1] + y*_roi_dim[0] +x;
    //                unsigned int idx1 = zz*_dim[0]*_dim[1] + yy*_dim[0] +xx;

    //                unsigned char l0 = _roi_cache.get()[idx0];
    //                unsigned char l1 = _mask_ref[idx1];
    //                if (l0 == 1)
    //                {
    //                    err_0 ++;
    //                    //std::cout << "ERROR 1 \n";
    //                }
    //                if (l0 == 0 && l1 != 0)
    //                {
    //                    err_1++; // Here is isolated point not error
    //                }
    //                if(l0 != 0 && l1 == 0)
    //                {
    //                    err_2 ++;
    //                    //std::cout << "ERROR 1\n";
    //                }
    //            }
    //        }
    //    }
    //    std::cout << "err : " << err_0 << " " << err_1 << " " << err_2 << std::endl;
    //}
    //////////////////////////////////////////////////////////////////////////

    if (tmp_cd.empty())
    {
        return;
    }

    //Calculate precise connect number
    for (auto it = tmp_cd.begin() ; it != tmp_cd.end() ; ++it)
    {
        it->second = 0;//clear number(has repeated idx)
    }

    for (unsigned int i = 0 ;i< _roi_dim[0]*_roi_dim[1]*_roi_dim[2] ; ++i)
    {
        unsigned char l = _roi_cache.get()[i];
        if (l != 0)
        {
            auto it = tmp_cd.find(l);
            if(it != tmp_cd.end())
            {
                it->second += 1;
            }
        }
    }

    std::cout << "connect doman : \n";
    for (auto it = tmp_cd.begin() ; it != tmp_cd.end() ; ++it)
    {
        std::cout << "label : " << int(it->first)  << "  voxel number : " << it->second << std::endl;
    }

    unsigned char major_label = tmp_cd.begin()->first;
    unsigned int max_num = tmp_cd.begin()->second;
    for (auto it = ++tmp_cd.begin() ; it!= tmp_cd.end() ; ++it)
    {
        if (it->second > max_num)
        {
            max_num = it->second;
            major_label = it->first;
        }
    }

    unsigned int curidx = 0;
    for (unsigned int z = 0 ; z< _roi_dim[2]; ++z)
    {
        for (unsigned int y = 0 ; y < _roi_dim[1] ; ++y)
        {
            for (unsigned int x = 0 ; x < _roi_dim[0] ; ++x)
            {
                const unsigned int zz = z + _min[2];
                const unsigned int yy = y + _min[1];
                const unsigned int xx = x + _min[0];
                const unsigned int idx0 = z*_roi_dim[0]*_roi_dim[1] + y*_roi_dim[0] +x;
                const unsigned int idx1 = zz*_dim[0]*_dim[1] + yy*_dim[0] +xx;

                const unsigned char l = _roi_cache.get()[idx0] ;
                if (l == major_label)
                {
                    _mask_ref[idx1] = _target_label;
                    ++curidx;
                }
                else
                {
                    _mask_ref[idx1] = 0;
                }

            }
        }
    }

    assert(curidx == max_num);
    std::cout << "major connected domain : " << major_label<< "   voxel number :" << max_num << std::endl; 


    //////////////////////////////////////////////////////////////////////////
    //Test
    //std::cout << "roi dim : " << _roi_dim[0] << " " << _roi_dim[1] << " " << _roi_dim[2] << std::endl;
    //std::ofstream out("D:/temp/roi_cd.raw" , std::ios::binary | std::ios::out);
    //if (out.is_open())
    //{
    //    out.write((char*)_roi_cache.get() , _roi_dim[0]*_roi_dim[1]*_roi_dim[2]);
    //    out.close();
    //}
    //////////////////////////////////////////////////////////////////////////
}



MED_IMAGING_END_NAMESPACE
