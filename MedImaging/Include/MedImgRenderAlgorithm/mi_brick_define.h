#ifndef MED_IMAGING_BRICK_UNIT_H
#define MED_IMAGING_BRICK_UNIT_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgArithmetic/mi_vector3f.h"

#include "boost/format.hpp"
#include "boost/tokenizer.hpp"
#include "boost/algorithm/string.hpp"  

MED_IMAGING_BEGIN_NAMESPACE

struct BrickCorner
{
    unsigned int min[3];
};

struct BrickUnit
{
    void* data;

    BrickUnit():data(nullptr)
    {}

    ~BrickUnit()
    {
        if (nullptr != data)
        {
            delete [] data;
            data = nullptr;
        }
    }
};

struct VolumeBrickInfo
{
    float min;
    float max;
};

struct MaskBrickInfo
{
    int label;
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

    bool operator < (const LabelKey& lk)
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


RenderAlgo_Export bool operator <(const LabelKey& left, const LabelKey& right);

MED_IMAGING_END_NAMESPACE

#endif