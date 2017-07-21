#ifndef MED_IMG_BRICK_UNIT_H
#define MED_IMG_BRICK_UNIT_H

#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgArithmetic/mi_vector3f.h"

#include "boost/format.hpp"
#include "boost/tokenizer.hpp"
#include "boost/algorithm/string.hpp"  

MED_IMG_BEGIN_NAMESPACE


/// \ Brick mesh structure
struct BrickEleIndex
{
    unsigned int idx[36];
};

struct BrickGeometry
{
    BrickEleIndex *brick_idx_units;
    float *vertex_array;//x y z
    float *color_array;//RGBA
    int vertex_count;

    BrickGeometry():brick_idx_units(nullptr),vertex_array(nullptr),color_array(nullptr),vertex_count(0)
    {
    }

    ~BrickGeometry()
    {
        if (nullptr != brick_idx_units)
        {
            delete [] brick_idx_units;
            brick_idx_units = nullptr;
        }

        if (nullptr != vertex_array)
        {
            delete [] vertex_array;
            vertex_array = nullptr;
        }

        if (nullptr != color_array)
        {
            delete [] color_array;
            color_array = nullptr;
        }
    }
};

/// \ Brick corner image coordinate
struct BrickCorner
{
    unsigned int min[3];
};

/// \ Brick info
struct VolumeBrickInfo
{
    float min;
    float max;
};

struct MaskBrickInfo
{
    int label;//0 for empty ; 255 for more than two different labels ; others for certain label
};

struct LabelKey
{
    std::string key;

    LabelKey():key("Empty")
    {
    }

    LabelKey(const std::vector<unsigned char>& labels)
    {
        std::vector<unsigned char> v = labels;
        //std::sort(v.begin() , v.end() , std::less<unsigned char>());
        std::sort(v.begin() , v.end());
        std::stringstream ss;
        for (auto it = v.begin() ; it != v.end() ; ++it)
        {
            if (*it != 0)
            {
                ss << (int)(*it )<<'|';
            }
        }
        std::string label = ss.str();
        if (label.empty())
        {
            label = "Empty";
        }

        key = label;
    }

    std::vector<unsigned char> ExtractLabels() const
    {
        if (key == "Empty")
        {
            return std::vector<unsigned char>();
        }
        else
        {
            std::vector<std::string> labels_string;  
            boost::split( labels_string, key , boost::is_any_of( "|") ); 
            if (labels_string.empty())
            {
                return std::vector<unsigned char>();
            }
            else
            {
                std::vector<unsigned char> labels;
                for (auto it = labels_string.begin() ; it != labels_string.end() ; ++it)
                {
                    if (!(*it).empty())
                    {
                        labels.push_back( (unsigned char)atoi((*it).c_str()));
                    }
                }
                return labels;
            }
        }
    }

    bool operator < (const LabelKey& lk) const
    {
        return this->key < lk.key;
    }
};

struct BrickDistance
{
    unsigned int id;
    float distance;

    BrickDistance():id(0),distance(0)
    {}
};


//RenderAlgo_Export bool operator <(const LabelKey& left, const LabelKey& right);

MED_IMG_END_NAMESPACE

#endif