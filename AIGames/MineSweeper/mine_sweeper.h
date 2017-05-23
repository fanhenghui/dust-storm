#pragma once

#include <memory>
#include <vector>
#include "Core/point2.h"
#include "Core/vector2.h"
#include "Core/point3.h"
#include "Core/vector3.h"

#include "mine.h"
#include "genetic_alg.h"

class NeuralNet;
class Mine;
class GLTexture2D;

class MineSweeper
{
public:
    MineSweeper();
    ~MineSweeper();

    int update(const std::vector<Mine> &mines);

    Vector2 get_closest_mine(const std::vector<Mine> &mines);

    int grab_mine(const std::vector<Mine> &mines);

    Point2 get_position() const;

    void increment_fitness(double val);

    double get_fitness() const;

    void draw();

    void set_texture(std::shared_ptr<GLTexture2D> tex_loser , std::shared_ptr<GLTexture2D> tex_elite);

    Chromosome to_chromosome();

    void set_weights(const std::vector<double>& weights);

    std::vector<int> get_weight_splits() const;

    void i_am_elite(bool flag);

    void reset();

private:
    void load_brain_i();

private:
    //input is closest mine (direction and magnitude)    and    current direction
    std::shared_ptr<NeuralNet> _brain;

    Point2 _position;
    Vector2 _direction;

    double _rotation;
    double _speed;

    //m_lTrack and m_rTrack store the current frame¡¯s output from the network. These are
    //the values that determine the mine sweeper¡¯s velocity and rotation
    double _left_track;
    double _right_track;

    double _fitness;

    //index position of closest mine
    int _closest_mine_id;

    //for draw
    std::shared_ptr<GLTexture2D> _tex_loser;
    std::shared_ptr<GLTexture2D> _tex_elite;
    bool _is_elite;

};