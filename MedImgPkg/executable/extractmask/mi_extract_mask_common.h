#ifndef MED_IMG_EXTRACT_MASK_COMMON_H
#define MED_IMG_EXTRACT_MASK_COMMON_H

#include <vector>
#include "arithmetic/mi_aabb.h"
#include "arithmetic/mi_point3.h"

using namespace medical_imaging;

struct Nodule {
    int type;//0 for unblinded-nodule ; 1 for non-nodule
    std::string name;
    std::vector<Point3>
    _points;//if size is 1:  for nodule means diameter < 3mm , for non-nodule means diameter > 3mm
    AABBI _aabb;
    int flag;//for indicate this nodule has been extracted : -1 under custom's confidence(skip) ; 0 not-anaylisis others  ; others makslabel

    Nodule(): type(0), flag(0) {

    }
};

#endif