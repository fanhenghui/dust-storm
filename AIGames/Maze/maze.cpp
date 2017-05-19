#include "maze.h"
#include <iostream>

Maze::Maze(int* array, int row, int column)
{
    _array.reset(new int[row*column]);
    memcpy(_array.get(), array, row*column * sizeof(int));
    _row = row;
    _column = column;

    _entry[0] = -1;
    _entry[1] = -1;
    _exit[0] = -1;
    _exit[1] = -1;

    for (int i = 0; i < _row; ++i)
    {
        for (int j = 0; j < _column; ++j)
        {
            int unit = _array[i*_column + j];
            if (ENTRY == unit)
            {
                _entry[0] = i;
                _entry[1] = j;
            }
            else if (EXIT == unit)
            {
                _exit[0] = i;
                _exit[1] = j;
            }
        }
    }

}

Maze::~Maze()
{

}

bool Maze::valid()
{
    return true;
}

void Maze::print()
{
    for (int i = 0 ; i<_row ; ++i)
    {
        for (int j = 0 ; j<_column ; ++j)
        {
            std::cout << _array[i*_column + j]<< " ";
        }
        std::cout << std::endl;
    }
}

bool Maze::test_route(const std::vector<int>& route , int& end_row, int& end_column)
{
    if (_exit[0] == -1 || _entry[0] == -1)
    {
        return false;
    }

    int step = 0;
    int cur_row = _entry[0];
    int cur_column = _entry[1];
    int cur_unit = 0;
    for (int i = 0 ; i< route.size() ; ++i)
    {
        step = route[i];
        switch (step)
        {
        case NORTH:
        {
            cur_row -= 1;
            if (cur_row < 0)
            {
                cur_row = 0;
            }
            else
            {
                cur_unit = get_unit(cur_row , cur_column);
                if (cur_unit == WALL)
                {
                    cur_row += 1;
                }
                else if (cur_unit == EXIT)
                {
                    end_row = cur_row;
                    end_column= cur_column;
                    return true;
                }
            }
            break;
        }
        case SOUTH:
        {
            cur_row += 1;
            if (cur_row > cur_row-1)
            {
                cur_row = cur_row-1;
            }
            else
            {
                cur_unit = get_unit(cur_row, cur_column);
                if (cur_unit == WALL)
                {
                    cur_row -= 1;
                }
                else if (cur_unit == EXIT)
                {
                    end_row = cur_row;
                    end_column = cur_column;
                    return true;
                }
            }
            break;
        }
        case EAST:
        {
            cur_column += 1;
            if (cur_column > cur_column - 1)
            {
                cur_column = cur_column - 1;
            }
            else
            {
                cur_unit = get_unit(cur_row, cur_column);
                if (cur_unit == WALL)
                {
                    cur_column -= 1;
                }
                else if (cur_unit == EXIT)
                {
                    end_row = cur_row;
                    end_column = cur_column;
                    return true;
                }
            }
            break;
        }
        default://WEST
        {
            cur_column -= 1;
            if (cur_column <0)
            {
                cur_column = 0;
            }
            else
            {
                cur_unit = get_unit(cur_row, cur_column);
                if (cur_unit == WALL)
                {
                    cur_column += 1;
                }
                else if (cur_unit == EXIT)
                {
                    end_row = cur_row;
                    end_column = cur_column;
                    return true;
                }
            }
            break;
        }
        }
    }

    end_row = cur_row;
    end_column = cur_column;
    return false;
}

void Maze::get_size(int& row, int &column)
{
    row = _row;
    column = _column;
}

int Maze::get_unit(int row, int column)
{
    return _array[row*_column + column];
}

void Maze::get_entry(int &row, int &column)
{
    row = _entry[0];
    column = _entry[1];
}

void Maze::get_exit(int &row, int &column)
{
    row = _exit[0];
    column = _exit[1];
}

int* Maze::get_maze()
{
    return _array.get();
}
