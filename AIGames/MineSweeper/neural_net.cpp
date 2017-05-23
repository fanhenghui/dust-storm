#include "neural_net.h"
#include <cassert>

#include "Core/common.h"

NeuralNet::NeuralNet(int input_num, int output_num, int hidden_layer_num , int neurons_per_hidden_layer , double bias , double sigmoid_response):
    _input_num(input_num),
    _output_num(output_num),
    _hidden_layer_num(hidden_layer_num),
    _neurons_per_hidden_layer(neurons_per_hidden_layer),
    _bias(bias),
    _sigmod_respond(sigmoid_response)
{
    create_net();
}

NeuralNet::~NeuralNet()
{

}

void NeuralNet::create_net()
{
    if (_hidden_layer_num > 0)
    {
        //first hidden layer 
        _layers.push_back(NeuroLayer(_neurons_per_hidden_layer , _input_num));

        //inner hidden layer
        for(int i = 0 ; i< _hidden_layer_num -1 ; ++i)
        {
            _layers.push_back(NeuroLayer(_neurons_per_hidden_layer , _neurons_per_hidden_layer));
        }

        //out put
        _layers.push_back(NeuroLayer(_output_num, _neurons_per_hidden_layer));
    }
    else
    {
        _layers.push_back(NeuroLayer(_output_num, _input_num));
    }
}

std::vector<double> NeuralNet::get_weights() const
{
    int idx = 0;
    std::vector<double> weights;
    for (int i = 0 ; i <_hidden_layer_num + 1; ++i)
    {
        const NeuroLayer& layer = _layers[i];
        for (int j = 0 ; j < layer._neuro_num ; ++j)
        {
            const Neuron& neuron = layer._neurons[j];
            for (int k =0 ; k < neuron._input_num ; ++k)
            {
                weights.push_back(neuron._weights[k]);
            }
        }
    }

    return weights;
}

int NeuralNet::get_num_of_weights() const
{
    int idx = 0;
    std::vector<double> weights;
    for (int i = 0; i < _hidden_layer_num + 1; ++i)
    {
        const NeuroLayer& layer = _layers[i];
        for (int j = 0; j < layer._neuro_num; ++j)
        {
            const Neuron& neuron = layer._neurons[j];
            idx += neuron._input_num;
        }
    }

    return idx;
}

void NeuralNet::set_weights(const std::vector<double>& weights)
{
    int idx = 0;
    for (int i = 0; i < _hidden_layer_num + 1; ++i)
    {
        NeuroLayer& layer = _layers[i];
        for (int j = 0; j < layer._neuro_num; ++j)
        {
            Neuron& neuron = layer._neurons[j];
            for (int k = 0; k < neuron._input_num; ++k)
            {
                neuron._weights[k] = weights[idx++];
            }
        }
    }
}

std::vector<double> NeuralNet::update(std::vector<double>& inputs)
{
    std::vector<double> outputs;

    if (inputs.size() != _input_num)
    {
        return outputs;
    }

    //for each layer
    for (int i = 0 ; i< _hidden_layer_num+1 ; ++i)
    {
        if (i > 0)
        {
            inputs = outputs;
        }

        outputs.clear();

        for (int j = 0 ; j<_layers[i]._neuro_num;++j)
        {
            double netoutput = 0;
            const int cur_input = _layers[i]._neurons[j]._input_num;
            for (int k = 0 ; k<cur_input -1 ; ++k)
            {
                netoutput += _layers[i]._neurons[j]._weights[k] * inputs[k];
            }
            //add in the bias
            netoutput += _layers[i]._neurons[j]._weights[cur_input - 1] * _bias;

            outputs.push_back(sigmoid_i(netoutput));

        }
    }

    return outputs;
}

std::vector<int> NeuralNet::get_splits() const
{
    std::vector<int> splits;
    int idx = 0;
    splits.push_back(idx);
    for (int i = 0; i < _hidden_layer_num + 1; ++i)
    {
        const NeuroLayer& layer = _layers[i];
        for (int j = 0; j < layer._neuro_num; ++j)
        {
            const Neuron& neuron = layer._neurons[j];
            idx += neuron._input_num;
            splits.push_back(idx);
        }
    }

    return splits;
}

double NeuralNet::sigmoid_i(double activation)
{
    return 1.0 / (1 + exp(-activation / _sigmod_respond));
}
