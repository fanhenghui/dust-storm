#include <vector>
#include <memory>
#include <iostream>

#include "neural_net.h"
#include "param.h"

namespace
{
    double input_array[] = { 0,0,0,1,1,0,1,1 };
    std::vector<std::vector<double>> _set_input = 
    {
        std::vector<double>(input_array , input_array+2),
        std::vector<double>(input_array+2 , input_array + 4),
        std::vector<double>(input_array + 4 , input_array + 6),
        std::vector<double>(input_array + 6 , input_array + 8),
    };
    std::vector<std::vector<double>> _set_output =
    {
        std::vector<double>(1 , 0),
        std::vector<double>(1 , 1),
        std::vector<double>(1 , 1),
        std::vector<double>(1 , 0)
    };

    std::shared_ptr<NeuralNet> _neural_net;
}


//////////////////////////////////////////////////////////////////////////
//对于异或问题，如果w的初始化全为同一个值则很难收敛
//w初始化为随机值则可以收敛
//////////////////////////////////////////////////////////////////////////
void main_xor(int argc, char* argv[])
{
    _neural_net.reset(new NeuralNet(2 , 1 , 1 , 2 , -1 , 1 , 0.8));

    for (int i = 0; i< 10000; ++i)
    {
        _neural_net->network_training_epoch(_set_input, _set_output);
        if (_neural_net->get_error() < Param::_error_threshold)
        {
            std::cout << "Iteration " << i << std::endl;
            break;
        }

        /*std::cout << "Iteration " << i << " 's weights : ";
        std::vector<double> weights = _neural_net->get_weights();
        for (int j = 0 ; j < weights.size() ; ++j)
        {
            std::cout << weights[j] << " , ";
        }
        std::cout << std::endl;*/
    }

    for (int i = 0; i< 4 ; ++i)
    {
        std::cout << _set_input[i][0] << " xor " << _set_input[i][1] <<
            " = " << _neural_net->update(_set_input[i])[0] << std::endl;
    }
}