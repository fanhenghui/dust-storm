#pragma once

#include <memory>
#include <vector>

#define  ROAD 0
#define  WALL 1
#define  ENTRY 2
#define  EXIT 3

#define  NORTH 0
#define  SOUTH 1
#define  EAST 2
#define  WEST 3

class Maze
{
public:
    //left up is zero(just as you look like)
    //0 road 
    //1 wall 
    //2 entry 
    //3 exit
    Maze(int* array, int row, int column);
    ~Maze();

    bool valid();

    void print();

    //0 north(up)
    //1 south(down)
    //2 east (right)
    //3 west (left)
    bool test_route(const std::vector<int>& route , int& end_row , int& end_column);

    void get_size(int& row, int &column);

    int get_unit(int row, int column);

    void get_entry(int &row, int &column);

    void get_exit(int &row, int &column);

    int* get_maze();


private:
    std::unique_ptr<int[]> _array;
    int _row;
    int _column;
    int _entry[2];
    int _exit[2];
};

