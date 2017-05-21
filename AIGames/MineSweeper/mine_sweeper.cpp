#include "mine_sweeper.h"
#include "Ext/gl/glew.h"

#define BODY_WIDTH 0.2
#define GUN_BARREL_LEGTH 0.25
#define TRACK_LENGTH 

MineSweeper::MineSweeper()
{

}

MineSweeper::~MineSweeper()
{

}

bool MineSweeper::update(std::vector<Point2> &mines)
{
    //use neural net to get output
    return true;
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

    return closest_object;
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
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINES);

    glColor3d(0.2, 0.2, 0.2);
    glBegin(GL_QUADS);

    /*glVertex2d(_position.x - WIDTH*0.5, _position.y - WIDTH*0.5);
    glVertex2d(_position.x - WIDTH*0.5, _position.y + WIDTH*0.5);
    glVertex2d(_position.x + WIDTH*0.5, _position.y + WIDTH*0.5);
    glVertex2d(_position.x + WIDTH*0.5, _position.y - WIDTH*0.5);*/

    glEnd();

    glPopMatrix();
    glPopAttrib();
}

