#include "mine_sweeper.h"

#include <cassert>

#include "Ext/gl/glew.h"

#include "Core/gl_texture_2d.h"
#include "Core/gl_utils.h"
#include "Core/common_utils.h"
#include "Core/matrix4.h"
#include "Core/quat4.h"
#include "Core/vector3.h"

#include "neural_net.h"
#include "param.h"

#define WIDTH 0.06

MineSweeper::MineSweeper():
    _speed(0),
    _rotation(0),
    _left_track(0),
    _right_track(0),
    _closest_mine_id(-1)
{
    load_brain_i();

    _position = Point2(rand_clamped(), rand_clamped());
    _rotation = rand_double()*PI * 2.0;
    _direction.x = sin(_rotation);
    _direction.y = cos(_rotation);
    _direction.normalize();

    _fitness = 0;
    _is_elite = false;
}

MineSweeper::~MineSweeper()
{

}

void MineSweeper::reset()
{
    _position = Point2(rand_clamped(), rand_clamped());
    _rotation = rand_double()*PI * 2.0;
    _direction.x = sin(_rotation);
    _direction.y = cos(_rotation);
    _direction.normalize();

    _fitness = 0;
    _is_elite = false;
}

int MineSweeper::update(const std::vector<Mine> &mines)
{
    //use neural net to get output
    const Vector2 closest_mine = get_closest_mine(mines);
    std::vector<double> input(Param::_input_num);
    input[0] = closest_mine.x;
    input[1] = closest_mine.y;
    input[2] = _direction.x;
    input[3] = _direction.y;

    std::vector<double> output =  _brain->update(input);
    assert(output.size() == Param::_output_num);

    _left_track = output[0];
    _right_track = output[1];

    //Calculate new _rotation and speed
    //calculate steering forces
    double rot_force  = _left_track- _right_track;

    //clamp rotation
    clamp(rot_force, -Param::_max_turn_rate, Param::_max_turn_rate);

    _rotation += rot_force;

    _speed = (_left_track+ _right_track);


    //TODO maybe need mutex
    {
        //update direction
        _direction.x = sin(_rotation);
        _direction.y = cos(_rotation);
        _direction.normalize();

        //update position
        static const double S_SPEED_ADJUST = 0.004;
        _position += (_direction* _speed * S_SPEED_ADJUST);
    }

    //wrap around window limits
    if (_position.x > 1.0) _position.x = -1.0;
    if (_position.y > 1.0) _position.y = -1.0;

    if (_position.x < -1.0) _position.x = 1.0;
    if (_position.y < -1.0) _position.y = 1.0;

    //check if sweeper the mine
    return grab_mine(mines);
}

Vector2 MineSweeper::get_closest_mine(const std::vector<Mine> &mines)
{
    double closest_so_far = std::numeric_limits<double>::max();
    _closest_mine_id = -1;

    Vector2 closest_object(0, 0);

    //cycle through mines to find closest
    for (int i = 0; i < mines.size(); i++)
    {
        double len_to_object = (mines[i]._position - _position).magnitude();

        if (len_to_object < closest_so_far)
        {
            closest_so_far = len_to_object;
            _closest_mine_id = i;
        }
    }

    closest_object = _position - mines[_closest_mine_id]._position;
    closest_object.normalize();
    return closest_object;
}

int MineSweeper::grab_mine(const std::vector<Mine> &mines)
{
    Vector2 obj = _position - mines[_closest_mine_id]._position;
    const double dis = obj.magnitude();
    if (dis < Mine::S_WIDTH)
    {
        return _closest_mine_id;
    }
    else
    {
        return -1;
    }
}

Point2 MineSweeper::get_position() const
{
    return _position;
}

void MineSweeper::increment_fitness(double val)
{
    _fitness += val;
}

double MineSweeper::get_fitness() const
{
    return _fitness;
}

void MineSweeper::draw()
{
    CHECK_GL_ERROR;

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    const static Vector3 v0(0, 1, 0);
    Vector3 v1(_direction.x, _direction.y, 0);
    Quat4 quat(angle_between(v0, v1), v0.cross_product(v1));
    Matrix4 mat = make_translate(Vector3(_position.x, _position.y, 0))*quat.to_matrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(mat._m);

    if (_is_elite)
    {
        _tex_elite->bind();
    }
    else
    {
        _tex_loser->bind();
    }

    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);

    glTexCoord2d(0, 0);
    glVertex2d( - WIDTH*0.5, - WIDTH*0.5);

    glTexCoord2d(0, 1);
    glVertex2d(- WIDTH*0.5, WIDTH*0.5);

    glTexCoord2d(1, 1);
    glVertex2d(WIDTH*0.5, WIDTH*0.5);

    glTexCoord2d(1, 0);
    glVertex2d(WIDTH*0.5, - WIDTH*0.5);

    glEnd();

    glDisable(GL_TEXTURE_2D);

    glPopMatrix();
    glPopAttrib();

    CHECK_GL_ERROR;
}

void MineSweeper::set_texture(std::shared_ptr<GLTexture2D> tex_loser , std::shared_ptr<GLTexture2D> tex_elite)
{
    _tex_loser = tex_loser;
    _tex_elite = tex_elite;
}

Chromosome MineSweeper::to_chromosome()
{
    return Chromosome(_brain->get_weights(), _fitness);
}

void MineSweeper::set_weights(const std::vector<double>& weights)
{
    _brain->set_weights(weights);
}

std::vector<int> MineSweeper::get_weight_splits() const
{
    return _brain->get_splits();
}

void MineSweeper::i_am_elite(bool flag)
{
    _is_elite = true;
}

void MineSweeper::load_brain_i()
{
    _brain.reset(new NeuralNet(
        Param::_input_num ,
        Param::_output_num,
        Param::_hidden_layer_num ,
        Param::_neurons_per_hidden_layer,
        Param::_bias ,
        Param::_sigmoid_response));
}
