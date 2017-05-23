#pragma once


class Param
{
public:
    static int _input_num;
    static int _output_num;
    static int _hidden_layer_num;
    static int _neurons_per_hidden_layer;
    static double _bias;
    static double _sigmoid_response;

    static int _mines_num;
    static int _mine_sweeper_num;
    static double _max_turn_rate;
    static int _update_fps;

    static int _ticks_num;//sweeper do ticks_num jobs then goto GA to train a new brain

    static double _crossover_rate;
    static double _mutation_rate;
    static double _max_perturbation;//max perturbation of mutation which add to ancestor
    static int _elite_num;//elitism to show
    static int _copy_elite_num; //keep into next generation

    //Add weight splits to consider one neuron's weight as crossover unit(thus don't cut one neuron's weight when crossover)
    static int _neuron_as_crossover_unit;
    

public:
    Param();
    static void Update();
};