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
    static double _max_speed;
    static int _update_fps;

    static int _ticks_num;//sweeper do ticks_num jobs then goto GA to train a new brain

    static double _crossover_rate;
    static double _mutation_rate;
    static double _max_perturbation;//max perturbation of mutation which add to ancestor
    static int _elite_num;//elitism to show
    static int _copy_elite_num; //keep into next generation

    

public:
    Param();
    static void Update();
};