#pragma once

#include <memory>
#include <vector>
#include "Core/point2.h"
#include "Core/vector2.h"
#include "Core/point3.h"
#include "Core/vector3.h"

#include "mine.h"

class NeuralNet;
class Mine;
class MineSweeper
{
public:
    MineSweeper();
    ~MineSweeper();

    bool update(std::vector<Point2> &mines);

    Vector2 get_closest_mine(const std::vector<Mine> &mines);

    Point2 get_position() const;

    void increment_fitness(double val);

    double get_fitness() const;

    void draw();

private:

private:
    std::shared_ptr<NeuralNet> _nerual_net;

    Point2 _position;
    Vector2 _look_at;

    double _rotation;
    double _speed;

    //m_lTrack and m_rTrack store the current frame¡¯s output from the network. These are
    //the values that determine the minesweeper¡¯s velocity and rotation
    double _left_track;
    double _right_track;

    double _fitness;

    //index position of closest mine
    int _closest_mine_id;

};