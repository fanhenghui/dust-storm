#pragma once

//total number of built in patterns
#define NUM_PATTERNS        11
//how many vectors each pattern contains
#define NUM_VECTORS         12

class Param
{
public:
    static int _input_num;
    static int _output_num;
    static int _hidden_layer_num;
    static int _neurons_per_hidden_layer;
    static double _bias;
    static double _sigmoid_response;
    static double _learning_rate;
    static double _error_threshold;
    static double _match_tolerance;

    //prevent algorithm stuck in a local minima.
    //add a momentum in each weight updating
    static double _momentum;

    //prevent overfitting
    static double _max_noise_to_add;


public:
    Param();
    static void Update();
};