#pragma once

#include <vector>
#include "Core/point3.h"

std::vector<int> grab_permutation(int limit);



class TSPMap
{
public:
    TSPMap();
    ~TSPMap();

private:
    std::vector<Point3> _cities;
    int _city_num;
    double _
};
