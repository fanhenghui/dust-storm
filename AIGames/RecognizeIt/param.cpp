#include "param.h"

int Param::_input_num = NUM_VECTORS*2;

int Param::_output_num = NUM_PATTERNS;

int Param::_hidden_layer_num = 1;

int Param::_neurons_per_hidden_layer = 6;

double Param::_bias = -1;

double Param::_sigmoid_response = 1;

double Param::_learning_rate =1.0;

double Param::_error_threshold = 0.001;

double Param::_match_tolerance = 0.96;

double Param::_momentum = 0.0;

double Param::_max_noise_to_add = 0.1;

Param::Param()
{

}

void Param::Update()
{

}
