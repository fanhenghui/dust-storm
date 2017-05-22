#pragma once

#include <vector>

struct Chromosome
{
    std::vector<double> _weights;
    double _fitness;

    Chromosome(const std::vector<double>& w, double f) :_weights(w), _fitness(f)
    {}

    Chromosome(int weight_num)
    {
        _weights.resize(weight_num);
        _fitness = 0;
    }

    bool operator<(const Chromosome& ch)
    {
        return (this->_fitness < ch._fitness);
    }

};

class GeneticAlg
{
public:
    GeneticAlg(
        double cross_rate,
        double mutation_rate,
        double max_perturbation);

    ~GeneticAlg();

    std::vector<Chromosome> epoch(std::vector<Chromosome>& ancestor);

    int get_generation() const;

    unsigned int get_fitest_id()const;

    unsigned int roulette_wheel_selection(std::vector<Chromosome>& chromosomes) const;

private:
    void update_fitness_scores_i(std::vector<Chromosome>& ancestor);

    void crossover_i(
        const Chromosome& mum, const Chromosome& dad,
        Chromosome& baby0, Chromosome& baby1);

    void mutate_i(Chromosome& ch);

private:
    int _chromosome_length;

    double _crossover_rate;
    double _mutation_rate;
    double _max_perturbation;
    unsigned int _pop_size;

    int _generation;

    unsigned int _fittest_chromosome_id;
    double _best_fitness_score;
    double _total_fitness_score;

};