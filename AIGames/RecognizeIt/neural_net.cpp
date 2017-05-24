#include "neural_net.h"

#include<cassert>
#include <iostream>

#include "Core/common.h"
#include "param.h"

NeuralNet::NeuralNet(
    int input_num, 
    int output_num, 
    int hidden_layer_num, 
    int neurons_per_hidden_layer, 
    double bias, 
    double sigmoid_response,
    double learning_rate) :
    _input_num(input_num),
    _output_num(output_num),
    _hidden_layer_num(hidden_layer_num),
    _neurons_per_hidden_layer(neurons_per_hidden_layer),
    _bias(bias),
    _sigmod_respond(sigmoid_response),
    _learning_rate(learning_rate)
{
    create_net();
    calculate_num_of_weights_i();

    _delta_weight.resize(get_num_of_weights());
    for (int i = 0 ; i < _delta_weight.size() ; ++i)
    {
        _delta_weight[i] = 0;
    }
}

NeuralNet::~NeuralNet()
{

}

void NeuralNet::create_net()
{
    if (_hidden_layer_num > 0)
    {
        //first hidden layer 
        _layers.push_back(NeuroLayer(_neurons_per_hidden_layer, _input_num));

        //inner hidden layer
        for (int i = 0; i < _hidden_layer_num - 1; ++i)
        {
            _layers.push_back(NeuroLayer(_neurons_per_hidden_layer, _neurons_per_hidden_layer));
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
    for (int i = 0; i < _hidden_layer_num + 1; ++i)
    {
        const NeuroLayer& layer = _layers[i];
        for (int j = 0; j < layer._neuro_num; ++j)
        {
            const Neuron& neuron = layer._neurons[j];
            for (int k = 0; k < neuron._input_num; ++k)
            {
                weights.push_back(neuron._weights[k]);
            }
        }
    }

    return weights;
}

int NeuralNet::get_num_of_weights() const
{
    return _num_of_weights;
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

std::vector<double> NeuralNet::update(const std::vector<double>& inputs_ori)
{
    std::vector<double> outputs;

    if (inputs_ori.size() != _input_num)
    {
        return outputs;
    }

    //for each layer
    std::vector<double> inputs = inputs_ori;
    for (int i = 0; i < _hidden_layer_num + 1; ++i)
    {
        if (i > 0)
        {
            inputs = outputs;
        }

        outputs.clear();

        for (int j = 0; j < _layers[i]._neuro_num; ++j)
        {
            double netoutput = 0;
            const int cur_input = _layers[i]._neurons[j]._input_num;
            for (int k = 0; k < cur_input - 1; ++k)
            {
                netoutput += _layers[i]._neurons[j]._weights[k] * inputs[k];
            }
            //add in the bias
            netoutput += _layers[i]._neurons[j]._weights[cur_input - 1] * _bias;
            netoutput = sigmoid_i(netoutput);

            outputs.push_back(netoutput);
            _layers[i]._neurons[j]._activation = netoutput;
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

void NeuralNet::calculate_num_of_weights_i()
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

    _num_of_weights = idx;
}

void NeuralNet::network_training_epoch(
    const std::vector<std::vector<double>>& set_in,
    const std::vector<std::vector<double>>& set_out)
{
    std::vector<double>::iterator cur_weight;
    std::vector<Neuron>::iterator cur_neuron_out , cur_neuron_hidden;

    _error_sum = 0;

    assert(set_in.size() == set_out.size());
    for (size_t train_id = 0; train_id < set_in.size(); ++train_id)
    {
        int pre_weight = _num_of_weights;

        //1 Get output based on current network
        std::vector<double> input = set_in[train_id];

        //Add jittering
       /* for (int i = 0; i < input.size(); ++i)
        {
            input[i] += rand_double()*Param::_max_noise_to_add;
        }*/

        const std::vector<double>& target = set_out[train_id];
        assert(input.size() == _input_num);
        assert(target.size() == _output_num);

        std::vector<double> output = this->update(input);
        assert(output.size() == _output_num);

        //2 Adjust the weights for the output layers
        assert(_layers.size() >= 2);
        for (int k = 0; k < _output_num; ++k)
        {
            NeuroLayer& layer_out = _layers[_layers.size() - 1];
            NeuroLayer& layer_hidden = _layers[_layers.size() - 2];

            Neuron& neuron_out = layer_out._neurons[k];

            const double t = target[k];
            const double o = output[k];
            //calculate error E(k) = ( t(k) - O(k) ) *O(k) * ( 1-O(k) ) 
            // t(k) is k's target(training set output) 
            // O(k) is k's output calculated by current network
            const double err = (t - o)*o*(1 - o);

            _error_sum += (t - o)*(t - o);//Sum of the squared errors(SSE)

            neuron_out._error = err;

            //update output neurons weight based on : w(j,k) := w(j,k) + L * E(k) * O(j)
            //w(j,k) weight hidden layer neuron j to output layer neuron k
            //L learning rate
            //O(j) hidden layer j's neuron's output(activation)
            int cur_weight = pre_weight - neuron_out._weights.size();//For momentum
            for (int j = 0; j < neuron_out._weights.size() - 1; ++j)//!!!Attention no bias (bias's activation is always -1)
            {
                double delta = _learning_rate * err * layer_hidden._neurons[j]._activation ;
                neuron_out._weights[j] += delta + _delta_weight[cur_weight] * Param::_momentum;
                _delta_weight[cur_weight++] = delta;
            }
            double delta = _learning_rate*err*Param::_bias ;
            neuron_out._weights[neuron_out._input_num - 1] += delta + _delta_weight[cur_weight] * Param::_momentum;
            _delta_weight[cur_weight++] = delta;

            pre_weight -= neuron_out._weights.size();//move to next weights rear
        }

        //3 Adjust the weights for the hidden layers
        for (int hidden_id = _layers.size() - 2; hidden_id >= 0; --hidden_id)
        {
            NeuroLayer& layer_out = _layers[hidden_id + 1];
            NeuroLayer& layer_hidden = _layers[hidden_id];

            if (hidden_id == 0)//last hidden layer
            {
                for (int j = 0; j < layer_hidden._neurons.size(); ++j)
                {
                    Neuron& neuron_hidden = layer_hidden._neurons[j];

                    //calculate error E(j) = O(j) * ( 1-O(j) ) * Sigmoid<1~k>( E(k) * w(j,k) )
                    double err = 0;
                    for (int k = 0; k < layer_out._neurons.size(); ++k)
                    {
                        err += layer_out._neurons[k]._error * layer_out._neurons[k]._weights[j];
                    }
                    const double o = neuron_hidden._activation;
                    err *= (o*(1 - o));
                    neuron_hidden._error = err;

                    //update hidden neurons weight based on : w(i,j) := w(i,j) + L * E(j) * O(i)
                    int cur_weight = pre_weight - neuron_hidden._weights.size();//For momentum
                    for (int i = 0; i < neuron_hidden._weights.size() - 1; ++i)//!!!Attention again no bias (bias's activation is always -1)
                    {
                        double delta = _learning_rate*err*input[i];
                        neuron_hidden._weights[i] += delta + _delta_weight[cur_weight] *Param::_momentum;
                        _delta_weight[cur_weight++] = delta;
                    }
                    double delta = _learning_rate*err*Param::_bias;
                    neuron_hidden._weights[neuron_hidden._weights.size() - 1] += delta + _delta_weight[cur_weight] * Param::_momentum;
                    _delta_weight[cur_weight++] = delta;

                    pre_weight -= neuron_hidden._weights.size();//move to next weights rear
                }
            }
            else//mid hidden layer
            {
                NeuroLayer& layer_input = _layers[hidden_id - 1];


                for (int j = 0; j < layer_hidden._neurons.size(); ++j)
                {
                    Neuron& neuron_hidden = layer_hidden._neurons[j];

                    //calculate error E(j) = O(j) * ( 1-O(j) ) * Sigmoid<1~k>( E(k) * w(j,k) )
                    double err = 0;
                    for (int k = 0; k < layer_out._neurons.size(); ++k)
                    {
                        err += layer_out._neurons[k]._error * layer_out._neurons[k]._weights[j];
                    }
                    const double o = neuron_hidden._activation;
                    err *= (o*(1 - o));
                    neuron_hidden._error = err;

                    //update hidden neurons weight based on : w(i,j) := w(i,j) + L * E(j) * O(i)
                    assert(layer_input._neurons.size() + 1 == neuron_hidden._weights.size());
                    for (int i = 0; i < neuron_hidden._weights.size() - 1; ++i)//!!!Attention again no bias (bias's activation is always -1)
                    {
                        neuron_hidden._weights[i] += _learning_rate*err*layer_input._neurons[i]._activation;
                    }
                    neuron_hidden._weights[neuron_hidden._weights.size() - 1] += _learning_rate*err*Param::_bias;
                }
            }


        }
    }

    ++_epoch_num;
}
