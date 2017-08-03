#ifndef MED_IMG_BRICK_DEFINE_H
#define MED_IMG_BRICK_DEFINE_H

#include <vector>
#include <algorithm>
#include "MedImgRenderAlgorithm/mi_render_algo_export.h"
#include "MedImgUtil/mi_string_number_converter.h"

MED_IMG_BEGIN_NAMESPACE

/// \ Brick mesh structure
struct BrickEleIndex
{
    unsigned int idx[36];
};

struct BrickGeometry
{
    BrickEleIndex *brick_idx_units;//element
    float *vertex_array;//vertex x y z
    float *color_array;//color r g b a
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

    std::vector<unsigned char> extract_labels() const
    {
        return LabelKey::extract_labels(*this);
    }

    static std::vector<unsigned char> extract_labels(const LabelKey& label_key)
    {
        const std::string& key = label_key.key;
        if (key == "Empty")
        {
            return std::vector<unsigned char>();
        }
        else
        {
            StrNumConverter<unsigned char> conv;
            std::vector<unsigned char> labels;
            std::string tmp;
            tmp.reserve(3);
            for (size_t i = 0; i<key.size() ; ++i)
            {
                if (key[i] != '|')
                {
                    tmp.push_back(key[i]);
                }
                else if (!tmp.empty())
                {
                    labels.push_back(conv.to_num(tmp));
                    tmp.clear();
                }
            }

            return labels;
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

MED_IMG_END_NAMESPACE

#endif