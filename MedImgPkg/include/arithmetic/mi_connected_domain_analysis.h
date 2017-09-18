#ifndef MEDIMGARITHMETIC_CONNECTED_DOMAIN_ANALYSIS_H
#define MEDIMGARITHMETIC_CONNECTED_DOMAIN_ANALYSIS_H

#include "arithmetic/mi_arithmetic_export.h"
#include "arithmetic/mi_arithmetic_utils.h"
#include "arithmetic/mi_arithmetic_logger.h"
#include <set>
#include <stack>
#include <vector>
#include <memory>

MED_IMG_BEGIN_NAMESPACE

class Arithmetic_Export ConnectedDomainAnalysis {
    struct POS {
        unsigned int x;
        unsigned int y;
        unsigned int z;
        POS() {

        }
        POS(unsigned int xx , unsigned int yy , unsigned int zz): x(xx), y(yy), z(zz) {

        }
    };

public:
    ConnectedDomainAnalysis() {
        memset(_min , 0 , sizeof(_min));
        memset(_max , 0 , sizeof(_max));
        memset(_dim , 0 , sizeof(_dim));
        memset(_roi_dim, 0 , sizeof(_roi_dim));
        _mask_ref = nullptr;
    }

    ~ConnectedDomainAnalysis() {

    }

    void set_dim(const unsigned int (&dim)[3]) {
        memcpy(_dim , dim , 3 * sizeof(unsigned int));
    }

    void set_mask_ref(unsigned char* mask_array) {
        _mask_ref = mask_array;
    }

    void set_target_label(unsigned char label) {
        _target_label = label;
    }

    void set_roi(const unsigned int (&min)[3] , const unsigned int (&max)[3]) {
        for (int i = 0 ; i < 3 ; ++i) {
            if (min[i] >= max[i]) {
                MI_ARITHMETIC_LOG(MI_FATAL) << "invalid roi dimension in connected domain analysis." <<
                    "min : (" << min[0] << "," << min[1] << "," << min[2] << ")" <<
                    "max : (" << max[0] << "," << max[1] << "," << max[2] << ")";
                ARITHMETIC_THROW_EXCEPTION("invalid roi dimension");
            }

            _min[i] = min[i];
            _max[i] = max[i];
            _roi_dim[i] = max[i] - min[i];
        }

        _roi_cache.reset(new unsigned char[_roi_dim[0]*_roi_dim[1]*_roi_dim[2]]);

        for (unsigned int z = 0  ; z < _roi_dim[2] ; ++z) {
            for (unsigned int y = 0 ; y < _roi_dim[1] ; ++y) {
                int zz = z + _min[2];
                int yy = y + _min[1];
                memcpy(_roi_cache.get() + z * _roi_dim[0]*_roi_dim[1] + y * _roi_dim[0],
                       _mask_ref + zz * _dim[0]*_dim[1] + yy * _dim[0] + _min[0] , _roi_dim[0]);
            }
        }

        for (unsigned int i = 0; i < _roi_dim[0]*_roi_dim[1]*_roi_dim[2] ; ++i) {
            if (_roi_cache[i] == _target_label) {
                _roi_cache[i] = 1;
            } else {
                _roi_cache[i] = 0;
            }
        }

        //////////////////////////////////////////////////////////////////////////
        //Test
        //std::ofstream out("D:/temp/roi.raw" , std::ios::binary | std::ios::out);
        //if (out.is_open())
        //{
        //    out.write((char*)_roi_cache.get() , _roi_dim[0]*_roi_dim[1]*_roi_dim[2]);
        //    out.close();
        //}
        //////////////////////////////////////////////////////////////////////////

    };

    void get_27_n(const unsigned int x , const unsigned int y , const unsigned int z ,
                  std::stack<POS>& s , unsigned char label , unsigned int& add) {
        const unsigned int zzz = z + 2 > _roi_dim[2]  ? _roi_dim[2] : z + 2;
        const unsigned int yyy = y + 2 > _roi_dim[1] ? _roi_dim[1] : y + 2;
        const unsigned int xxx = x + 2 > _roi_dim[0] ? _roi_dim[0] : x + 2;

        add = 0;
        std::vector<unsigned int> pos_tmp;
        pos_tmp.reserve(26);

        for (unsigned int zz = z - 1 < 0 ? 0 : z - 1 ; zz < zzz ; ++zz) {
            for (unsigned int yy = y - 1 < 0 ? 0 : y - 1; yy < yyy ; ++yy) {
                for (unsigned int xx = x - 1 < 0 ? 0 : x - 1; xx < xxx ; ++xx) {
                    if (xx != x || yy != y || zz != z) {
                        unsigned int idx = zz * _roi_dim[0] * _roi_dim[1] + yy * _roi_dim[0] + xx;;

                        if (_roi_cache.get()[idx] == 1) {
                            s.push(POS(xx, yy, zz));
                            pos_tmp.push_back(idx);
                        }
                    }
                }
            }
        }

        add += static_cast<unsigned int>(pos_tmp.size());

        for (auto it = pos_tmp.begin() ; it != pos_tmp.end() ; ++it) {
            _roi_cache.get()[*it] = label;
        }
    }

    void get_result(std::vector<std::vector<unsigned int>>& cd);

    void get_result_major(std::vector<unsigned int>& cd);

    void keep_major();

    void seed_filling(std::stack<POS>& s , unsigned char label , unsigned int& num);

protected:
private:
    unsigned int _dim[3];
    unsigned int _min[3];
    unsigned int _max[3];
    unsigned int _roi_dim[3];
    unsigned char* _mask_ref;
    unsigned char _target_label;
    std::unique_ptr<unsigned char[]> _roi_cache;
};

MED_IMG_END_NAMESPACE
#endif