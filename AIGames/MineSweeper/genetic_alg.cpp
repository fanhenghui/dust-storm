#include "genetic_alg.h"

#include <iostream>
#include <algorithm>

#include "Core/common_utils.h"
#include "param.h"

GeneticAlg::GeneticAlg(double cross_rate,
    double mutation_rate,
    double max_perturbation) :
    _crossover_rate(cross_rate),
    _mutation_rate(mutation_rate),
    _max_perturbation(max_perturbation),
    _generation(0),
    _fittest_chromosome_id(-1),
    _best_fitness_score(std::numeric_limits<double>::min()),
    _total_fitness_score(0)
{

}

GeneticAlg::~GeneticAlg()
{

}

std::vector<Chromosome> GeneticAlg::epoch( std::vector<Chromosome>& ancestor)
{
    _pop_size = static_cast<unsigned int>(ancestor.size());
    _chromosome_length = static_cast<int>(ancestor[0]._weights.size());

    std::sort(ancestor.begin(), ancestor.end());

    update_fitness_scores_i(ancestor);

    std::cout << "Fittest score " << _best_fitness_score << std::endl;

    std::vector<Chromosome> next_generation;
    next_generation.reserve(_pop_size);

    //Keep elitism
    for (int i = 0 ; i<Param::_copy_elite_num; ++i)
    {
        next_generation.push_back(ancestor[_pop_size-1-i]);
        next_generation[i]._fitness = 0;
    }

    unsigned int cur_pop = static_cast<unsigned int>(Param::_copy_elite_num);

    while (cur_pop < _pop_size)
    {
        const unsigned int mum_id = roulette_wheel_selection(ancestor);
        const unsigned int dad_id = roulette_wheel_selection(ancestor);
        const Chromosome& mum = ancestor[mum_id];
        const Chromosome& dad = ancestor[dad_id];

        Chromosome baby0(_chromosome_length);
        Chromosome baby1(_chromosome_length);

        crossover_i(mum, dad, baby0, baby1);

        mutate_i(baby0);
        mutate_i(baby1);

        next_generation.push_back(baby0);
        next_generation.push_back(baby1);

        cur_pop += 2;
    }

    ++_generation;
    return next_generation;
}

int GeneticAlg::get_generation() const
{
    return _generation;
}

double GeneticAlg::get_fittest_score() const
{
    return _best_fitness_score;
}

double GeneticAlg::get_average_score() const
{
    return _total_fitness_score / _pop_size;
}

unsigned int GeneticAlg::roulette_wheel_selection(std::vector<Chromosome>& chromosomes) const
{
    const double slice = rand_double() * _total_fitness_score;
    double current_slice = 0;
    unsigned int i;
    for (i = 0; i < _pop_size; ++i)
    {
        current_slice += chromosomes[i]._fitness;
        if (current_slice > slice)
        {
            return i;
        }
    }

    return i - 1;
}

void GeneticAlg::set_crossover_splits(const std::vector<int>& splits)
{
    _crossover_splits = splits;
}

void GeneticAlg::update_fitness_scores_i(std::vector<Chromosome>& chromosomes)
{
    //Get fittest
    double max_fitness = chromosomes[0]._fitness;
    unsigned int fittest_id = 0;
    double total_scores = 0;
    for (unsigned int i = 1; i < _pop_size; ++i)
    {
        total_scores += chromosomes[i]._fitness;
        if (chromosomes[i]._fitness > max_fitness)
        {
            max_fitness = chromosomes[i]._fitness;
            fittest_id = i;
        }
    }

    _best_fitness_score = max_fitness;
    _fittest_chromosome_id = fittest_id;
    _total_fitness_score = total_scores;
}

void GeneticAlg::crossover_i(const Chromosome& mum, const Chromosome& dad, Chromosome& baby0, Chromosome& baby1)
{
    if (_crossover_splits.empty())
    {
        if (rand_double() > _crossover_rate || (mum._weights == dad._weights))
        {
            baby0._weights = mum._weights;
            baby1._weights = dad._weights;
            return;
        }
        else
        {
            int cp = rand_int(0, _chromosome_length - 1);
            for (int i = 0; i < cp; ++i)
            {
                baby0._weights[i] = mum._weights[i];
                baby1._weights[i] = dad._weights[i];
            }
            for (int i = cp; i < _chromosome_length; ++i)
            {
                baby0._weights[i] = dad._weights[i];
                baby1._weights[i] = mum._weights[i];
            }
        }
    }
    else
    {
        crossover_splits_i(mum, dad, baby0, baby1);
    }
    
}

void GeneticAlg::crossover_splits_i(
    const Chromosome& mum, 
    const Chromosome& dad, 
    Chromosome& baby0, 
    Chromosome& baby1)
{
    if (rand_double() > _crossover_rate || (mum._weights == dad._weights))
    {
        baby0._weights = mum._weights;
        baby1._weights = dad._weights;
        return;
    }

    const int sp1 = rand_int(0, _crossover_splits.size() - 2);
    const int sp2 = rand_int(sp1, _crossover_splits.size() - 1);
    const int cp1 = _crossover_splits[sp1];
    const int cp2 = _crossover_splits[sp2];

    for (int i = 0; i < cp1; ++i)
    {
        baby0._weights[i] = mum._weights[i];
        baby1._weights[i] = dad._weights[i];
    }
    for (int i = cp1; i < cp2; ++i)
    {
        baby0._weights[i] = dad._weights[i];
        baby1._weights[i] = mum._weights[i];
    }
    for (int i = cp2; i < _chromosome_length; ++i)
    {
        baby0._weights[i] = mum._weights[i];
        baby1._weights[i] = dad._weights[i];
    }
}

void GeneticAlg::mutate_i(Chromosome& ch)
{
    for (int i = 0; i < _chromosome_length; ++i)
    {
        if (rand_double() < _mutation_rate)
        {
            ch._weights[i] += rand_clamped() * _max_perturbation;
        }
    }
}
