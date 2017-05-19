#include "maze_solver.h"
#include "maze.h"

MazeSolver::MazeSolver(
    double cross_rate, 
    double mutation_rate, 
    unsigned int pop_size,
    int chromosome_length,
    int gene_length):
    _cross_over_rate(cross_rate),
    _mutation_rate(mutation_rate),
    _pop_size(pop_size),
    _chromosome_length(chromosome_length),
    _gene_length(gene_length),
    _generation(0),
    _fittest_chromosome_id(-1),
    _best_fitness_score(std::numeric_limits<double>::min()),
    _total_fitness_score(0)
{
    create_start_population();
}

MazeSolver::~MazeSolver()
{

}

void MazeSolver::set_maze(std::shared_ptr<Maze> maze)
{
    _maze = maze;
}

void MazeSolver::epoch()
{
    update_fitness_scores_i();

    std::vector<Chromosome> next_generation;
    next_generation.reserve(_pop_size);
    unsigned int cur_pop = 0;

    while (cur_pop < _pop_size)
    {
        const unsigned int mum_id = roulette_wheel_selection();
        const unsigned int dad_id= roulette_wheel_selection();
        Chromosome& mum = _chromosomes[mum_id];
        Chromosome& dad = _chromosomes[dad_id];

        Chromosome baby0(_chromosome_length , false);
        Chromosome baby1(_chromosome_length, false);

        crossover_i(mum, dad, baby0, baby1);

        mutate_i(baby0);
        mutate_i(baby1);

        next_generation.push_back(baby0);
        next_generation.push_back(baby1);

        cur_pop += 2;
    }

    ++_generation;
    _chromosomes = next_generation;

}

int MazeSolver::get_generation() const
{
    return _generation;
}

const MazeSolver::Chromosome& MazeSolver::get_fittest_chromosome() const
{
    return _chromosomes[_fittest_chromosome_id];
}

unsigned int MazeSolver::roulette_wheel_selection() const
{
    const double slice = rand_double() * _total_fitness_score;
    double current_slice = 0;
    unsigned int i;
    for (i = 0 ; i < _pop_size ; ++i)
    {
        current_slice += _chromosomes[i]._fitness;
        if(current_slice > slice)
        { 
            return i;
        }
    }

    return i - 1;
}

void MazeSolver::create_start_population()
{
    _chromosomes.clear();
    _chromosomes.reserve(_pop_size);
    for (unsigned int i = 0 ; i<_pop_size ; ++i)
    {
        _chromosomes.push_back(Chromosome(_chromosome_length));
    }
    _generation = 0;
}

void MazeSolver::update_fitness_scores_i()
{
    int exit_row(0), exit_column(0);
    _maze->get_exit(exit_row, exit_column);

    //Update all population
    //multi-thread
    for (unsigned int i = 0 ; i<_pop_size ; ++i)
    {
        Chromosome& chrom = _chromosomes[i];
        std::vector<int> route = decode(chrom);
        int end_row(0), end_column(0);
        _maze->test_route(route, end_row , end_column);
        int diff_v = abs(end_row - exit_row);
        int diff_h = abs(end_column - exit_column);
        double score = 1 / static_cast<double>(diff_v + diff_h + 1);
        chrom._fitness = score;
    }

    //Get fittest
    double max_fitness = _chromosomes[0]._fitness;
    unsigned int fittest_id = 0;
    double total_scores = 0;
    for (unsigned int i = 1; i<_pop_size ; ++i)
    {
        total_scores += _chromosomes[i]._fitness;
        if (_chromosomes[i]._fitness > max_fitness)
        {
            max_fitness = _chromosomes[i]._fitness;
            fittest_id = i;
        }
    }

    _best_fitness_score = max_fitness;
    _fittest_chromosome_id = fittest_id;
    _total_fitness_score = total_scores;

}

void MazeSolver::crossover_i(
    const Chromosome& mum, const Chromosome& dad,
    Chromosome& baby0, Chromosome& baby1)
{
    if (rand_double() > _cross_over_rate)
    {
        baby0._bits = mum._bits;
        baby1._bits = dad._bits;
    }
    else
    {
        int cp = rand_int(0, _chromosome_length - 1);
        for (int i = 0 ; i<cp ; ++i)
        {
            baby0._bits[i] = mum._bits[i];
            baby1._bits[i] = dad._bits[i];
        }
        for (int i = cp; i < _chromosome_length; ++i)
        {
            baby0._bits[i] = dad._bits[i];
            baby1._bits[i] = mum._bits[i];
        }
    }
}

void MazeSolver::mutate_i(Chromosome& ch)
{
    if (rand_double() > _mutation_rate)
    {
        return;
    }
    else
    {
        int cp = rand_int(0, _chromosome_length - 1);
        for (int i = cp ; i < _chromosome_length ; ++i)
        {
            ch._bits[i] = 1 - ch._bits[i];
        }
    }
}

std::vector<int> MazeSolver::decode(const Chromosome& ch)
{
    std::vector<int> result(ch._num_bits / 2);
    for (int i = 0 ; i < result.size() ; ++i)
    {
        result[i] = ( (ch._bits[i * 2]) << 1 )+ ch._bits[i * 2 + 1];
    }
    return std::move(result);
}
