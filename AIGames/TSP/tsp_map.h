#pragma once

#include <vector>
#include "Core//common_utils.h"
#include "Core/point3.h"


//circular cites
class TSPMap
{
public:
    TSPMap(double radius , int city_num);
    ~TSPMap();

    const std::vector<Point3>& get_cities() const;

    double get_radius() const;

    double get_distance(const std::vector<int>& tour) const;

    double get_best_possible_route() const;

private:
    void calculate_cities_circular_i();

private:
    std::vector<Point3> _cities;
    int _city_num;
    double _radius;
    double _best_possible_route;
};
