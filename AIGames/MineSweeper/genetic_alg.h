//#pragma once
//
//#include <vector>
//
//class GeneticAlg
//{
//public:
//    struct Chromosome
//    {
//        std::vector<double> _weights;
//        double _fitness;
//
//        Chromosome(const std::vector<double>& w , double f):_weights(w) , _fitness(f)
//        {}
//
//        bool operator<(const Chromosome& ch)
//        {
//            return (this->_fitness< ch._fitness);
//        }
//
//    };
//
//public:
//    GeneticAlg(
//        double corss_rate,
//        double mutation_rate,
//        unsigned int pop_size,
//        int weight_num,
//        int gene_length);
//
//    ~GeneticAlg();
//
//    void epoch();
//
//    int get_generation() const;
//
//    void create_start_population();
//
//    unsigned int roulette_wheel_selection() const;
//
//    const Chromosome& get_fittest_chromosome() const;
//
//private:
//    void update_fitness_scores_i();
//
//    void crossover_i(
//        const Chromosome& mum, const Chromosome& dad,
//        Chromosome& baby0, Chromosome& baby1);
//
//    void mutate_i(Chromosome& ch);
//
//private:
//    std::vector<Chromosome> _chromosomes;
//    int _chromosome_length;
//
//    double _crossover_rate;
//    double _mutation_rate;
//    unsigned int _pop_size;
//
//    int _generation;
//
//    unsigned int _fittest_chromosome_id;
//    double _best_fitness_score;
//    double _total_fitness_score;
//
//};