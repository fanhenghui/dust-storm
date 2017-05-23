#pragma once

#include <vector>
#include <map>

class GestureData
{
public:
    GestureData();
    ~GestureData();

    void get_train_set(
        std::vector<std::vector<double>>&set_in ,
        std::vector<std::vector<double>>&set_out);

    std::string get_pattern_name(int id);
protected:
private:
    std::vector <std::vector <double>> _set_in;
    std::vector <std::vector <double>> _set_out;
    std::map<int, std::string> _pattern_name;
};