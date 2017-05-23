#pragma once

#include <vector>
#include "Core/point2.h"


class GestureDrawer
{
public:
    GestureDrawer(int splits = 12);
    ~GestureDrawer();

    void push_back(const Point2& pt);

    void clear();

    const std::vector<Point2>& get_splits() const { return _split_points; };

    void calculate_splits();

    void draw();

private:
    std::vector<Point2> _points;
    int _split_num;
    double _route_length;
    std::vector<double> _split_length;
    std::vector<Point2> _split_points;
};