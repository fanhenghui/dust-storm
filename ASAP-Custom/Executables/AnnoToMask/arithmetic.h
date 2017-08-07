#ifndef ARITHMETIC_H
#define ARITHMETIC_H

#include <iostream>
#include <fstream>

template<class T>
class AABB
{
public:
    T _min[2];//Lower Left Back
    T _max[2];//Upper Right Front

public:
    AABB() {};
    virtual ~AABB() {};

    bool operator == (const AABB& aabb) const
    {
        return (_min[0] == aabb._min[0] && _min[1] == aabb._min[1] &&
            _max[0] == aabb._max[0] && _max[1] == aabb._max[1] );
    }

    bool operator != (const AABB& aabb) const
    {
        return (_min[0] != aabb._min[0] || _min[1] != aabb._min[1] ||
            _max[0] != aabb._max[0] || _max[1] != aabb._max[1]);
    }

    void add(int x, int y)
    {
        _min[0] = _min[0] <= x ? _min[0] : x;
        _min[1] = _min[1] <= y ? _min[1] : y;

        _max[0] = _max[0] >= x ? _max[0] : x;
        _max[1] = _max[1] >= y ? _max[1] : y;
    }
};

template<class T>
bool aabb_to_aabb_cross(const AABB<T>& l, const AABB<T>& r, AABB<T>& result)
{
    if ((l._min[0] > r._max[0] || l._min[1] > r._max[1] ) ||
        (r._min[0] > l._max[0] || r._min[1] > l._max[1] ))
    {
        return false;
    }

    for (int i = 0; i < 2; ++i)
    {
        result._max[i] = (std::min)(l._max[i], r._max[i]);
        result._min[i] = (std::max)(l._min[i], r._min[i]);
    }

    return true;
}

template<class T>
bool aabb_to_aabb_combine(const AABB<T>& l, const AABB<T>& r, AABB<T>& result)
{
    if ((l._min[0] > r._max[0] || l._min[1] > r._max[1]) ||
        (r._min[0] > l._max[0] || r._min[1] > l._max[1]))
    {
        return false;
    }

    for (int i = 0; i < 2; ++i)
    {
        result._max[i] = (std::max)(l._max[i], r._max[i]);
        result._min[i] = (std::min)(l._min[i], r._min[i]);
    }

    return true;
}

int write_raw(const std::string& path, char* buffer, unsigned int length)
{
    if (nullptr == buffer || path.empty()) {
        return -1;
    }

    std::ofstream out(path.c_str(), std::ios::out | std::ios::binary);

    if (!out.is_open()) {
        return -1;
    }


    out.write(buffer, length);
    out.close();

    return 0;
}

#endif