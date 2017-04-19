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
    unsigned int m_Min[3];
};

struct BrickUnit
{
    void* m_pData;

    BrickUnit():m_pData(nullptr)
    {}
    ~BrickUnit()
    {
        if (nullptr != m_pData)
        {
            delete [] m_pData;
            m_pData = nullptr;
        }
    }
};

struct VolumeBrickInfo
{
    float m_fMin;
    float m_fMax;
};

struct MaskBrickInfo
{
    int m_iLabel;
};

struct LabelKey
{
    std::string m_sKey;

    LabelKey():m_sKey("Empty")
    {
    }

    LabelKey(const std::vector<unsigned char>& vecLabels)
    {
        std::vector<unsigned char> v = vecLabels;
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
        std::string sLabel = ss.str();
        if (sLabel.empty())
        {
            sLabel = "Empty";
        }

        m_sKey = sLabel;
    }

    std::vector<unsigned char> ExtractLabels() const
    {
        if (m_sKey == "Empty")
        {
            return std::vector<unsigned char>();
        }
        else
        {
            std::vector<std::string> sLabels;  
            boost::split( sLabels, m_sKey , boost::is_any_of( "|") ); 
            if (sLabels.empty())
            {
                return std::vector<unsigned char>();
            }
            else
            {
                std::vector<unsigned char> vecLabel;
                for (auto it = sLabels.begin() ; it != sLabels.end() ; ++it)
                {
                    if (!(*it).empty())
                    {
                        vecLabel.push_back( (unsigned char)atoi((*it).c_str()));
                    }
                }
                return vecLabel;
            }
        }
    }

    bool operator < (const LabelKey& lk)
    {
        return this->m_sKey < lk.m_sKey;
    }
};

struct BrickDistance
{
    unsigned int m_id;
    float m_fDistance;

    BrickDistance():m_id(0),m_fDistance(0)
    {}
};


RenderAlgo_Export bool operator <(const LabelKey& _Left, const LabelKey& _Right);

MED_IMAGING_END_NAMESPACE

#endif