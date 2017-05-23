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
    static double _learning_rate;


public:
    Param();
    static void Update();
};