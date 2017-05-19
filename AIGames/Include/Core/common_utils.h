#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <string>
#include <sstream>

template<typename T>
std::string num_to_string(T num)
{
    std::stringstream ss;
    ss << num;
    return ss.str();
}

template<typename T>
std::string num_to_string_decimal(T num, int precision)
{
    std::stringstream ss;
    ss << std::setprecision(precision) << std::fixed << num;
    return ss.str();
}

template<typename T>
T string_to_num(const std::string& s)
{
    std::stringstream ss(s);
    T v;
    ss >> v;
    return v;
}

inline int rand_int(int x, int y) { return rand() % (y - x + 1) + x; }

inline double rand_double() { return (rand()) / (RAND_MAX + 1.0); }

inline bool rand_bool() { return 1 == rand_int(0, 1);}

#endif