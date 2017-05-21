#pragma once

#include <vector>
#include <memory>
#include "Core/common_utils.h"

std::vector<int> grab_permutation(int limit);

struct Chromosome
{
    std::vector<int> _city_tours;
    double _fitness;
    int _city_num;

    Chromosome(int num, bool create_start_tours = true) :_city_num(num), _fitness(0)
    {
        if (create_start_tours)
        {
            _city_tours = grab_permutation(num);
        }
        else
        {
            _city_tours.resize(num);
        }
    }
};

class TSPMap;
class TSPSolver
{
public:
    TSPSolver(double corss_rate,
        double mutation_rate,
        unsigned int pop_size,
        int city_num);

    ~TSPSolver();

    void set_tsp_map(std::shared_ptr<TSPMap> tsp_map);

    void epoch();

    int get_generation() const;

    void create_start_population();

    unsigned int roulette_wheel_selection() const;

    const Chromosome& get_fittest_chromosome() const;

    double get_shortest_route() const;

private:
    void update_fitness_scores_i();

    void crossover_i(
        const Chromosome& mum, const Chromosome& dad,
        Chromosome& baby0, Chromosome& baby1);

    void mutate_i(Chromosome& ch);

    bool check_chromosome_valid(const Chromosome& ch);

private:
    //////////////////////////////////////////////////////////////////////////
    //crossover operator
    //permutation crossover operator(PMX)
    void crossover_partially_mapped_crossover_i(
        const Chromosome& mum, const Chromosome& dad,
        Chromosome& baby0, Chromosome& baby1);

    void crossover_order_based_crossover_i(
        const Chromosome& mum, const Chromosome& dad,
        Chromosome& baby0, Chromosome& baby1);

    //暂时没调通
    void crossover_position_based_crossover_i(
        const Chromosome& mum, const Chromosome& dad, 
        Chromosome& baby0, Chromosome& baby1);

    //////////////////////////////////////////////////////////////////////////
    //Mutation operator
    //Exchange (EM)
    void mutate_exchange_i(Chromosome& ch);

    //Scramble (SM)
    void mutate_scramble_i(Chromosome& ch);

    void mutate_displacement_i(Chromosome& ch);

    void mutate_insertion_i(Chromosome& ch);


private:
    std::vector<Chromosome> _chromosomes;
    int _chromosome_length;

    double _crossover_rate;
    double _mutation_rate;
    unsigned int _pop_size;

    int _generation;

    double _total_fitness_score;
    double _shortest_route;
    double _longest_route;

    unsigned int _shortest_route_id;

    std::shared_ptr<TSPMap> _tsp_map;
    std::vector<unsigned int> _loser;
    std::vector<unsigned int> _winner;
};