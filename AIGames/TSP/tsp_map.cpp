#include "tsp_map.h"

TSPMap::TSPMap(double radius , int city_num):_radius(radius),_city_num(city_num), _best_possible_route(0)
{
    calculate_cities_circular_i();
}

TSPMap::~TSPMap()
{

}

const std::vector<Point3>& TSPMap::get_cities() const
{
    return _cities;
}

double TSPMap::get_radius() const
{
    return _radius;
}

double TSPMap::get_distance(const std::vector<int>& tour) const
{
    double dis = 0;
    for (int i = 0 ; i< tour.size()-1; ++i)
    {
        const Point3& pre = _cities[tour[i]];
        const Point3& cur= _cities[tour[i+1]];
        dis += (pre - cur).magnitude();
    }
    return dis;
}

double TSPMap::get_best_possible_route() const
{
    return _best_possible_route;
}

void TSPMap::calculate_cities_circular_i()
{
    const double slice = PI*2 / _city_num;
    _cities.resize(_city_num);
    for (int i = 0 ; i<_city_num ; ++i)
    {
        double degree = slice*i;
        _cities[i] = Point3(_radius*cos(degree), _radius*sin(degree), 0);
    }

    double dis_sum = 0;
    for (int i = 0; i < _city_num-1; ++i)
    {
        dis_sum += (_cities[i] - _cities[i + 1]).magnitude();
    }
    _best_possible_route = dis_sum;
}
