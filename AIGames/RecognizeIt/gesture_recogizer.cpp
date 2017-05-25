#include "gesture_recogizer.h"

#include <iostream>

#include "param.h"
#include "neural_net.h"
#include "gesture_data.h"

GestureRecogizer::GestureRecogizer()
{
    _brain.reset(new NeuralNet(
        Param::_input_num, 
        Param::_output_num, 
        Param::_hidden_layer_num, 
        Param::_neurons_per_hidden_layer, 
        Param::_bias, 
        Param::_sigmoid_response, 
        Param::_learning_rate));

    _data.reset(new GestureData());

}

GestureRecogizer::~GestureRecogizer()
{

}

std::string GestureRecogizer::get_gesture_pattern_name(std::vector<double>& input)
{
    std::vector<double> output = _brain->update(input);
    int idx = -1;
    double max_val = std::numeric_limits<double>::min();
    for (int i = 0 ; i<output.size() ; ++i)
    {
        if (output[i] > max_val && output[i] > Param::_match_tolerance)
        {
            max_val = output[i];
            idx = i;
        }
    }

    if (idx == -1)
    {
        return "Uncertain";
    }
    else
    {
        return _data->get_pattern_name(idx);
    }
}

void GestureRecogizer::train()
{
    std::vector<std::vector<double>> set_in;
    std::vector<std::vector<double>> set_out;
    _data->get_train_set(set_in, set_out);

    const int iteration = 50000;
    int i = 0;
    for ( ; i< iteration ; ++i)
    {
        _brain->network_training_epoch(set_in, set_out);
        std::cout << _brain->get_error() << std::endl;
        if (_brain->get_error() < Param::_error_threshold)
        {
            break;
        }
    }
    std::cout << "Iteration stop at " << i << "\n";

}
