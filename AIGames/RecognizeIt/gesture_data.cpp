#include "gesture_data.h"
#include "param.h"



const double InputVectors[NUM_PATTERNS][NUM_VECTORS * 2] =
{
    1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0,                                 //right

    -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0,                     //left

    0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1,                                 //down

    0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1,                     //up

    -1,0, -1,0, -1,0, 0,1, 0,1, 0,1, 1,0, 1,0, 1,0, 0,-1, 0,-1, 0,-1,                           //clockwise square

    1,0, 1,0, 1,0, 0,1, 0,1, 0,1, -1,0, -1,0, -1,0, 0,-1, 0,-1, 0,-1,                           //anticlockwise square

    1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, -0.45,0.9, -0.9, 0.45, -0.9,0.45,              //Right Arrow 

    -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, -1,0, 0.45,0.9, 0.9, 0.45, 0.9,0.45,        //Left Arrow

    -0.7,-0.7, -0.7,-0.7, -0.7,-0.7, -0.7,-0.7, -0.7,-0.7, -0.7,-0.7, -0.7,-0.7, -0.7,-0.7,             //south west
    -0.7,-0.7, -0.7,-0.7, -0.7,-0.7, -0.7,-0.7,

    0.7,-0.7, 0.7,-0.7, 0.7,-0.7, 0.7,-0.7, 0.7,-0.7, 0.7,-0.7, 0.7,-0.7, 0.7,-0.7, 0.7,-0.7,            //south east
    0.7,-0.7, 0.7,-0.7, 0.7,-0.7,

    //-0.7,0.7, -0.7,0.7, -0.7,0.7, -0.7,0.7, -0.7,0.7, -0.7,0.7, -0.7,0.7, -0.7,0.7,             //north west
    //-0.7,0.7, -0.7,0.7, -0.7,0.7, -0.7,0.7,

    //0.7,0.7, 0.7,0.7, 0.7,0.7, 0.7,0.7, 0.7,0.7, 0.7,0.7, 0.7,0.7, 0.7,0.7, 0.7,0.7,            //north east
    //0.7,0.7, 0.7,0.7, 0.7,0.7,

    1,0, 1,0, 1,0, 1,0, -0.72,0.69,-0.7,0.72,0.59,0.81, 1,0, 1,0, 1,0, 1,0, 1,0                 //zorro
};


const char* Names[NUM_PATTERNS] =
{
    "Right",
    "Left",
    "Down",
    "Up",
    "Clockwise Square",
    "Anti-Clockwise Square",
    "Right Arrow",
    "Left Arrow",
    "South West",
    "South East",
    //"North West",
    //"North East",
    "Zorro"
};

GestureData::GestureData()
{
    //for each const pattern  
    std::vector<std::vector<double> > patterns;
    for (int ptn = 0; ptn < NUM_PATTERNS; ++ptn)
    {
        //add it to the vector of patterns
        std::vector<double> temp;

        for (int v = 0; v < NUM_VECTORS * 2; ++v)
        {
            temp.push_back(InputVectors[ptn][v]);
        }
        patterns.push_back(temp);

        _pattern_name[ptn] = Names[ptn];
    }

    //empty the vectors
    _set_in.clear();
    _set_out.clear();

    //add each pattern
    for (int ptn = 0; ptn < NUM_PATTERNS; ++ptn)
    {
        //add the data to the training set
        _set_in.push_back(patterns[ptn]);

        //create the output vector for this pattern. First fill a 
        //std::vector with zeros
        std::vector<double> outputs(NUM_PATTERNS, 0);

        //set the relevant output to 1
        outputs[ptn] = 1;

        //add it to the output set
        _set_out.push_back(outputs);
    }
}

GestureData::~GestureData()
{

}

void GestureData::get_train_set(std::vector<std::vector<double>>&set_in, std::vector<std::vector<double>>&set_out)
{
    set_in = _set_in;
    set_out = _set_out;
}

std::string GestureData::get_pattern_name(int id)
{
    return _pattern_name[id];
}
