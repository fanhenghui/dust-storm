#include "tsp_solver.h"
#include <set>
#include <map>
#include <cassert>

#include "tsp_map.h"


#define LOSER_SCORE  0.3
#define WINNER_SCORE 0.98
#define NUM_BEST_TO_ADD 1

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

void choose_section(int& begin, int&end, int length, int min_span)
{
    int pos0 = rand_int(0, length - 1);
    int pos1 = pos0;
    while (abs(pos1 - pos0) < min_span)
    {
        pos1 = rand_int(0, length - 1);
    }
    begin = std::min(pos0, pos1);
    end = std::max(pos0, pos1);
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

    //for (int i = 0; i < _winner.size(); ++i)
    //{
    //    next_generation.push_back(_chromosomes[_winner[i]]);
    //}

    for (int i = 0; i < NUM_BEST_TO_ADD; ++i)
    {
        next_generation.push_back(_chromosomes[_shortest_route_id]);
    }

    unsigned int cur_pop = _winner.size() + NUM_BEST_TO_ADD;
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
    
    for (unsigned int i = 0; i<_pop_size ; ++i)
    {
        const double dis = _tsp_map->get_distance(_chromosomes[i]._city_tours);
        _chromosomes[i]._fitness = dis;

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

    //Using route length as fitness
    //double score_sum = 0;
    //for (unsigned int i = 0; i < _pop_size; ++i)
    //{
    //    _chromosomes[i]._fitness = longest_route - _chromosomes[i]._fitness;
    //    score_sum += _chromosomes[i]._fitness;
    //}

    //Using length recipr
    //_winner.clear();
    //_loser.clear();
    double score_sum = 0;
    for (unsigned int i = 0; i < _pop_size; ++i)
    {
        _chromosomes[i]._fitness = (longest_route + _tsp_map->get_best_possible_route() - _chromosomes[i]._fitness)/longest_route;
        score_sum += _chromosomes[i]._fitness;
        /*if (_chromosomes[i]._fitness< LOSER_SCORE)
        {
            _loser.push_back(i);
        }*/
        /*if (_chromosomes[i]._fitness > WINNER_SCORE)
        {
            _winner.push_back(i);
        }*/
    }

    //std::cout << "Loser than " << LOSER_SCORE << " num : " << _loser.size() << std::endl;

    //for (int i = 0 ; i < _loser.size() ; ++i)
    //{
    //    unsigned int loser = _loser[i];
    //    double fitness = _chromosomes[loser]._fitness;
    //    while (fitness < LOSER_SCORE)
    //    {
    //        _chromosomes[loser] = Chromosome(_chromosome_length);
    //        const double dis = longest_route + _tsp_map->get_best_possible_route() -  _tsp_map->get_distance(_chromosomes[loser]._city_tours);
    //        fitness = dis / longest_route;
    //    }
    //    _chromosomes[loser]._fitness = fitness;
    //}


    _longest_route = longest_route;
    _shortest_route = shortest_route;
    _shortest_route_id = shortest_id;
    _total_fitness_score = score_sum;
}

void TSPSolver::crossover_i(const Chromosome& mum, const Chromosome& dad, Chromosome& baby0, Chromosome& baby1)
{
    crossover_order_based_crossover_i(mum , dad , baby0 , baby1);
}

void TSPSolver::mutate_i(Chromosome& ch)
{
    mutate_insertion_i(ch);
}

bool TSPSolver::check_chromosome_valid(const Chromosome& ch)
{
    std::set<int> s;
    for (int i = 0; i < _chromosome_length; ++i)
    {
        s.insert(ch._city_tours[i]);
    }
    if (s.size() != _chromosome_length)
    {
        return false;
    }
    else
    {
        return true;
    }
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
       /* {
            if (check_chromosome_valid(baby0))
            {
                std::cout << "error\n";
            }
            if (check_chromosome_valid(baby1))
            {
                std::cout << "error\n";
            }
        }*/
    }
}

void TSPSolver::crossover_order_based_crossover_i(const Chromosome& mum, const Chromosome& dad, Chromosome& baby0, Chromosome& baby1)
{
    if (rand_double() > _crossover_rate || mum._city_tours == dad._city_tours)
    {
        baby0 = mum;
        baby1 = dad;
        return;
    }

    baby0 = mum;
    baby1 = dad;

    //holds the chosen cities
    std::vector<int> tempCities;

    //holds the positions of the chosen cities
    std::vector<int> positions;

    //first city position
    int Pos = rand_int(0, mum._city_tours.size() - 2);

    //keep adding random cities until we can add no more
    //record the positions as we go
    while (Pos < mum._city_tours.size())
    {
        positions.push_back(Pos);

        tempCities.push_back(mum._city_tours[Pos]);

        //next city
        Pos += rand_int(1, mum._city_tours.size() - Pos);
    }

    //so now we have n amount of cities from mum in the tempCities
    //vector we can impose their order in dad.
    int cPos = 0;

    for (int cit = 0; cit < baby1._city_tours.size(); ++cit)
    {
        for (int i = 0; i < tempCities.size(); ++i)
        {
            if (baby1._city_tours[cit] == tempCities[i])
            {
                baby1._city_tours[cit] = tempCities[cPos];

                ++cPos;

                break;
            }
        }
    }

    //now vice versa
    tempCities.clear();
    cPos = 0;

    //first grab the cities from the same positions in dad
    for (int i = 0; i < positions.size(); ++i)
    {
        tempCities.push_back(dad._city_tours[positions[i]]);
    }

    //and impose their order in mum
    for (int cit = 0; cit < baby0._city_tours.size(); ++cit)
    {
        for (int i = 0; i < tempCities.size(); ++i)
        {
            if (baby0._city_tours[cit] == tempCities[i])
            {
                baby0._city_tours[cit] = tempCities[cPos];

                ++cPos;

                break;
            }
        }
    }
}

bool test_num(const std::vector<int> &vec, const int &number)
{
    for (int i = 0; i < vec.size(); ++i)
    {
        if (vec[i] == number)
        {
            return true;
        }
    }

    return false;
}


void TSPSolver::crossover_position_based_crossover_i(const Chromosome& mum, const Chromosome& dad, Chromosome& baby1, Chromosome& baby2)
{
    if (rand_double() > _crossover_rate || mum._city_tours == dad._city_tours)
    {
        baby1 = mum;
        baby2 = dad;
        return;
    }

    //initialize the babies with minus values so we can tell which positions
    //have been filled later in the algorithm
    baby1._city_tours.assign(mum._city_tours.size(), -1);
    baby2._city_tours.assign(mum._city_tours.size(), -1);

    int l = baby2._city_tours.size();

    //holds the positions of the chosen cities
    std::vector<int> positions;

    //first city position
    int Pos = rand_int(0, mum._city_tours.size() - 2);

    //keep adding random cities until we can add no more
    //record the positions as we go
    while (Pos < mum._city_tours.size())
    {
        positions.push_back(Pos);

        //next city
        Pos += rand_int(1, mum._city_tours.size() - Pos);
    }

    //now we have chosen some cities it's time to copy the selected cities
    //over into the offspring in the same position.
    for (int pos = 0; pos < positions.size(); ++pos)
    {
        //baby1 receives from mum
        baby1._city_tours[positions[pos]] = mum._city_tours[positions[pos]];

        //baby2 receives from dad
        baby2._city_tours[positions[pos]] = dad._city_tours[positions[pos]];
    }

    //fill in the blanks. First create two position markers so we know
    //whereabouts we are in baby1 and baby2
    int c1 = 0, c2 = 0;

    for (int pos = 0; pos < mum._city_tours.size(); ++pos)
    {
        //advance position marker until we reach a free position
        //in baby2
        while ((baby2._city_tours[c2] > -1) && (c2 < mum._city_tours.size()))
        {
            ++c2;
        }

        //baby2 gets the next city from mum which is not already
        //present
        if ((!test_num(baby2._city_tours, mum._city_tours[pos])))
        {
            baby2._city_tours[c2] = mum._city_tours[pos];
        }

        //now do the same for baby1
        while ((baby1._city_tours[c1] > -1) && (c1 < mum._city_tours.size()))
        {
            ++c1;
        }

        //baby1 gets the next city from dad which is not already
        //present
        if ((!test_num(baby1._city_tours, dad._city_tours[pos])))
        {
            baby1._city_tours[c1] = dad._city_tours[pos];
        }
    }
}

void TSPSolver::mutate_exchange_i(Chromosome& ch)
{
    if (rand_double() > _mutation_rate)
    {
        return;
    }

    int exchange_num = 1;
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

void TSPSolver::mutate_scramble_i(Chromosome& ch)
{
    if (rand_double() > _mutation_rate)
    {
        return;
    }

    const int min_span = 3;
    int begin,end;
    choose_section(begin , end , _chromosome_length , min_span);

    int span = end - begin;
    while (span > 0)
    {
        int p0 = rand_int(begin, end);
        int p1 = p0;
        while (p1 == p0)
        {
            p1 = rand_int(begin, end);
        }
        std::swap(ch._city_tours[p0] , ch._city_tours[p1]);
        --span;
    }
}

void TSPSolver::mutate_displacement_i(Chromosome& ch)
{
    if (rand_double() > _mutation_rate)
    {
        return;
    }

    const int min_span = 3;
    int beg, end;
    choose_section(beg, end, _chromosome_length, min_span);

    //setup iterators for our beg/end points
    std::vector<int>::iterator SectionStart = ch._city_tours.begin() + beg;
    std::vector<int>::iterator SectionEnd = ch._city_tours.begin() + end;

    //hold on to the section we are moving
    std::vector<int> TheSection;
    TheSection.assign(SectionStart, SectionEnd);

    //erase from current position
    ch._city_tours.erase(SectionStart, SectionEnd);

    //move an iterator to a random insertion location
    std::vector<int>::iterator curPos;
    curPos = ch._city_tours.begin() + rand_int(0, ch._city_tours.size() - 1);

    //re-insert the section
    ch._city_tours.insert(curPos, TheSection.begin(), TheSection.end());
}

void TSPSolver::mutate_insertion_i(Chromosome& ch)
{
    if (rand_double() > _mutation_rate)
    {
        return;
    }

    //create an iterator for us to work 
    std::vector<int>::iterator pos_it = ch._city_tours.begin() + rand_int(0, _chromosome_length - 1);
    //keep a note of the genes value
    int city_num = *pos_it;
    //remove from the chromosome
    ch._city_tours.erase(pos_it);
    //move the iterator to the insertion location
    pos_it = ch._city_tours.begin() + rand_int(0, _chromosome_length - 1);
    ch._city_tours.insert(pos_it , city_num);
}
