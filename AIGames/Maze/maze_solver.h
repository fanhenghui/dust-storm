#pragma once

#include <memory>
#include <vector>
#include "Core/common_utils.h"


class Maze;
class MazeSolver
{
public:
    struct Chromosome
    {
        std::vector<int> _bits;
        int _num_bits;
        double _fitness;

        Chromosome(int num_bits, bool create_start_bits = true) :
            _fitness(0), _num_bits(num_bits)
        {
            _bits.resize(num_bits);
            if (create_start_bits)
            {
                for (int i = 0; i < num_bits; ++i)
                {
                    _bits[i] = rand_int(0, 1);
                }
            }
        }
    };

public:
    MazeSolver(
        double corss_rate,
        double mutation_rate,
        unsigned int pop_size,
        int chromosome_length,
        int gene_length);

    ~MazeSolver();

    void set_maze(std::shared_ptr<Maze> maze);

    void epoch();//generation loop

    void create_start_population();

    int get_generation() const;

    const Chromosome& get_fittest_chromosome() const;

    unsigned int roulette_wheel_selection() const;

    static std::vector<int> decode(const Chromosome& ch);

private:
    void update_fitness_scores_i();

    void crossover_i(const Chromosome& mum, const Chromosome& dad ,
        Chromosome& baby0 , Chromosome& baby1);

    void mutate_i(Chromosome& ch);
private:
    std::vector<Chromosome> _chromosomes;
    unsigned int _pop_size;
    int _chromosome_length;
    int _gene_length;

    double _cross_over_rate;
    double _mutation_rate;

    unsigned int _fittest_chromosome_id;
    double _best_fitness_score;
    double _total_fitness_score;

    int _generation;

    std::shared_ptr<Maze> _maze;

};