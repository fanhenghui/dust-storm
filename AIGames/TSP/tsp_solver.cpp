#include "tsp_solver.h"
#include <set>
#include <map>
#include "tsp_map.h"


#define NUM_BEST_TO_ADD 2

std::vector<int> grab_permutation(int limit)
{
    std::set<int> s;
    std::vector<int> v;
    v.reserve(limit);
    while (s.size() != limit)
    {
        int id = rand_int(0, limit - 1);
        if (s.find(id) == s.end())
        {
            s.insert(id);
            v.push_back(id);
        }
    }
    return std::move(v);
}

TSPSolver::TSPSolver(double corss_rate, double mutation_rate, unsigned int pop_size, int city_num):
    _crossover_rate(corss_rate),
    _mutation_rate(mutation_rate),
    _pop_size(pop_size),
    _chromosome_length(city_num),
    _shortest_route_id(0),
    _generation(0),
    _shortest_route(std::numeric_limits<double>::max()),
    _longest_route(std::numeric_limits<double>::min()),
    _total_fitness_score(0)
{
    
}

TSPSolver::~TSPSolver()
{

}

void TSPSolver::set_tsp_map(std::shared_ptr<TSPMap> tsp_map)
{
    _tsp_map = tsp_map;
}

void TSPSolver::epoch()
{
    update_fitness_scores_i();

    if (_shortest_route < _tsp_map->get_best_possible_route())
    {
        return;
    }

    std::vector<Chromosome> next_generation;

    for (int i = 0; i < NUM_BEST_TO_ADD; ++i)
    {
        next_generation.push_back(_chromosomes[_shortest_route_id]);
    }

    unsigned int cur_pop = NUM_BEST_TO_ADD;
    while (cur_pop < _pop_size)
    {
        cur_pop += 2;

        const unsigned int mum_id = roulette_wheel_selection();
        const unsigned int dad_id = roulette_wheel_selection();

        Chromosome& mum = _chromosomes[mum_id];
        Chromosome& dad = _chromosomes[dad_id];

        Chromosome baby0(_chromosome_length , false);
        Chromosome baby1(_chromosome_length , false);

        crossover_i(mum, dad, baby0, baby1);

        mutate_i(baby0);
        mutate_i(baby1);

        next_generation.push_back(baby0);
        next_generation.push_back(baby1);
    }

    _chromosomes = next_generation;
    ++_generation;
}

int TSPSolver::get_generation() const
{
    return _generation;
}

void TSPSolver::create_start_population()
{
    _generation = 0;
    _chromosomes.clear();
    for (unsigned int i = 0 ; i<_pop_size ; ++i)
    {
        _chromosomes.push_back(Chromosome(_chromosome_length));
    }
}

unsigned int TSPSolver::roulette_wheel_selection() const
{
    const double slice = rand_double() * _total_fitness_score;
    double current_slice = 0;
    unsigned int i;
    for (i = 0; i < _pop_size; ++i)
    {
        current_slice += _chromosomes[i]._fitness;
        if (current_slice > slice)
        {
            return i;
        }
    }

    return i - 1;
}

const Chromosome& TSPSolver::get_fittest_chromosome() const
{
    return _chromosomes[_shortest_route_id];
}

double TSPSolver::get_shortest_route() const
{
    return _shortest_route;
}

void TSPSolver::update_fitness_scores_i()
{
    double longest_route = std::numeric_limits < double>::min();
    double shortest_route = std::numeric_limits < double>::max();
    unsigned int shortest_id = -1;
    double score_sum = 0;
    for (unsigned int i = 0; i<_pop_size ; ++i)
    {
        const double dis = _tsp_map->get_distance(_chromosomes[i]._city_tours);
        _chromosomes[i]._fitness = dis;
        score_sum += dis;

        if (dis < shortest_route)
        {
            shortest_route = dis;
            shortest_id = i;
        }
        if (dis > longest_route)
        {
            longest_route = dis;
        }
    }

    for (unsigned int i = 0; i < _pop_size; ++i)
    {
        _chromosomes[i]._fitness = longest_route - _chromosomes[i]._fitness;
    }

    _longest_route = longest_route;
    _shortest_route = shortest_route;
    _shortest_route_id = shortest_id;
    _total_fitness_score = score_sum;
}

void TSPSolver::crossover_i(const Chromosome& mum, const Chromosome& dad, Chromosome& baby0, Chromosome& baby1)
{
    crossover_partially_mapped_crossover_i(mum , dad , baby0 , baby1);
}

void TSPSolver::mutate_i(Chromosome& ch)
{
    mutate_exchange_i(ch);
}

void TSPSolver::crossover_partially_mapped_crossover_i(
    const Chromosome& mum, const Chromosome& dad, Chromosome& baby0, Chromosome& baby1)
{
    if (rand_double() > _crossover_rate || mum._city_tours == dad._city_tours)
    {
        baby0 = mum;
        baby1 = dad;
        return;
    }
    else
    {
        //find begin and end
        int begin = rand_int(0, _chromosome_length / 2);
        int end = begin;
        while (end <= begin)
        {
            end = rand_int(0, _chromosome_length - 1);
        }

        //create map
        std::map<int, int> mapped;
        for (int i = 0 ; i< _chromosome_length ; ++i)
        {
            mapped[i] = i;
        }

        std::vector<int> l0;
        l0.reserve(end - begin + 1);
        std::vector<int> l1;
        l0.reserve(end - begin + 1);
        for (int i = begin ; i <end+1 ; ++i)
        {
            l0.push_back(mum._city_tours[i]);
            l1.push_back(dad._city_tours[i]);
        }

        //crossover
        baby0 = mum;
        baby1 = dad;
        for (int i = 0; i<end-begin+1 ; ++i)
        {
            for (int j = 0; j < _chromosome_length; ++j)
            {
                if (baby0._city_tours[j] == l0[i])
                {
                    baby0._city_tours[j] = l1[i];
                }
                else if (baby0._city_tours[j] == l1[i])
                {
                    baby0._city_tours[j] = l0[i];
                }

                if (baby1._city_tours[j] == l0[i])
                {
                    baby1._city_tours[j] = l1[i];
                }
                else if (baby1._city_tours[j] == l1[i])
                {
                    baby1._city_tours[j] = l0[i];
                }
            }
        }
        

        //////////////////////////////////////////////////////////////////////////
        //test
        /*{
            std::set<int> s;
            for (int i = 0 ; i<_chromosome_length ; ++i)
            {
                s.insert(baby0._city_tours[i]);
            }
            if (s.size() != _chromosome_length)
            {
                std::cout << "error\n";
            }

            for (int i = 0; i<_chromosome_length; ++i)
            {
                s.insert(baby0._city_tours[i]);
            }
            if (s.size() != _chromosome_length)
            {
                std::cout << "error\n";
            }
        }*/
    }
}

void TSPSolver::mutate_exchange_i(Chromosome& ch)
{
    if (rand_double() > _mutation_rate)
    {
        return;
    }

    int exchange_num =  rand_int(0, _chromosome_length - 1);
    int cur_exchange = 0;
    while (cur_exchange < exchange_num)
    {
        int pos0 = rand_int(0, _chromosome_length-1);
        int pos1 = pos0;
        while (pos1 == pos0)
        {
            pos1 = rand_int(0, _chromosome_length-1);
        }

        std::swap(ch._city_tours[pos0], ch._city_tours[pos1]);

        ++cur_exchange;
    }
}
