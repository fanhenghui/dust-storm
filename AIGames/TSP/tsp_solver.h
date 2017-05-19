#pragma once

#include <vector>

std::vector<int> grab_permutation(int limit);

struct Chromosome
{
    std::vector<int> _city_tours;
    double _fitness;
    int _city_num;

    Chromosome(int num) :_city_num(num), _fitness(0)
    {
        _city_tours = grab_permutation(num);
    }
};

class TSPSolver
{
public:
protected:
private:
};