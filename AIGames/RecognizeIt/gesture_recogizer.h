#pragma once

#include <memory>
#include <vector>

class NeuralNet;
class GestureData;
class GestureRecogizer
{
public:
    GestureRecogizer();
    ~GestureRecogizer();

    std::string get_gesture_pattern_name(std::vector<double>& input);

    void train();


private:
    std::shared_ptr<NeuralNet> _brain;
    std::shared_ptr<GestureData> _data;
};