#pragma once

#include <vector>
#include "Core/common_utils.h"

struct Neuron
{
    int _input_num;
    std::vector<double> _weights;

    double _activation;
    double _error;


    Neuron(int input_num) :_input_num(input_num + 1)// bias
    {
        _weights.reserve(_input_num);
        for (int i = 0; i < _input_num; ++i)
        {
            _weights.push_back(rand_clamped());
        }
    }
};

struct NeuroLayer
{
    int _neuro_num;
    std::vector<Neuron> _neurons;

    NeuroLayer(int neuro_num, int per_neuro_input_num) :_neuro_num(neuro_num)
    {
        _neurons.reserve(_neuro_num);
        for (int i = 0; i < neuro_num; ++i)
        {
            _neurons.push_back(Neuron(per_neuro_input_num));
        }
    }
};

class NeuralNet
{
public:
    NeuralNet(
        int input_num, 
        int output_num, 
        int hidden_layer_num, 
        int neurons_per_hidden_layer, 
        double bias, 
        double sigmoid_response,
        double learning_rate);
    ~NeuralNet();

    void create_net();

    std::vector<double> get_weights() const;

    void set_weights(const std::vector<double>& weights);

    int get_num_of_weights() const;

    std::vector<double> update(const std::vector<double>& inputs);

    std::vector<int> get_splits() const;

    //given a training set this method performs one iteration of the
    //backpropagation algorithm. The training sets comprise of series
    //of vector inputs and a series of expected vector outputs.
    void network_training_epoch(
        const std::vector<std::vector<double>>& set_in, 
        const std::vector<std::vector<double>>& set_out);

    int get_epoch()const { return _epoch_num; };

    bool is_trained() const { return _is_trained; };

    double get_error() const { return _error_sum; };

private:
    double sigmoid_i(double activation);

    void calculate_num_of_weights_i();


private:
    int _input_num;
    int _output_num;
    int _hidden_layer_num;
    int _neurons_per_hidden_layer;

    double _bias;
    double _sigmod_respond;

    std::vector<NeuroLayer> _layers;

    //For backpropagation
    double _learning_rate;
    double _error_sum;
    bool _is_trained;
    int _epoch_num;

    //momentum
    int _num_of_weights;
    std::vector<double> _delta_weight;

};