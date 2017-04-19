#ifndef MED_IMAGING_CORNER_INFO_UNIT_H
#define MED_IMAGING_CORNER_INFO_UNIT_H

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgArithmetic/mi_color_unit.h"

MED_IMAGING_BEGIN_NAMESPACE

enum CornerPos
{
    LB,// left bottom
    RB,//right bottom
    RT,// right top
    LT,//left top
};

struct CornerInfoUnit 
{
    CornerInfoUnit():m_sContext("NULL"),m_ePos(LB),m_eColor(255,255,255)
    {

    }
    std::string m_sContext;
    CornerPos m_ePos;
    RGBUnit m_eColor;
};

MED_IMAGING_END_NAMESPACE


#endif