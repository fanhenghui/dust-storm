#ifndef MED_IMAGING_RAY_CASTING_BRICK_ACC_H_
#define MED_IMAGING_RAY_CASTING_BRICK_ACC_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgRenderAlgorithm/mi_brick_define.h"

#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgArithmetic/mi_vector4f.h"
#include "MedImgArithmetic/mi_sampler.h"
#include "MedImgArithmetic/mi_matrix4f.h"
#include "MedImgArithmetic/mi_matrix4.h"

MED_IMAGING_BEGIN_NAMESPACE

class RayCaster;

class RayCastingCPUBrickAcc
{
public:
    RayCastingCPUBrickAcc(std::shared_ptr<RayCaster> pRayCaster);
    ~RayCastingCPUBrickAcc();
    void render(int iTestCode = 0);

    //////////////////////////////////////////////////////////////////////////
    //For testing 
    const std::vector<BrickDistance>& get_brick_distance() const;
    unsigned int get_ray_casting_brick_count() const;
public:

private:
    void sort_brick_i();//Just for orthogonal camera(same ray direction)

    void ray_casting_in_brick_i(unsigned int id ,  const std::shared_ptr<RayCaster>& pRayCaster);

private:
    std::weak_ptr<RayCaster> m_pRayCaster;
    //Cache
    int _width;
    int _height;
    Vector4f* m_pEntryPoints;
    Vector4f* m_pExitPoints;
    unsigned int _dim[3];
    RGBAUnit* m_pColorCanvas;

    //Brick struct
    unsigned int m_uiBrickDim[3];
    unsigned int m_uiBrickSize;
    unsigned int m_uiBrickExpand;
    unsigned int m_uiBrickCount;
    BrickCorner* m_pBrickCorner;
    BrickUnit* m_pVolumeBrickUnit;
    VolumeBrickInfo* m_pVolumeBrickInfo;
    BrickUnit* m_pMaskBrickUnit;
    MaskBrickInfo* m_pMaskBrickInfo;
    Matrix4f m_matMVP;
    Matrix4f m_matMVPInv;
    Matrix4 m_matMVPInv0;


    //Brick cache
    std::vector<BrickDistance> m_vecBrickCenterDistance;
    unsigned int m_uiInterBrickNum;
    std::unique_ptr<float[]> m_pRayResult;
    int m_iRayCount;
    Vector3f m_vRayDirNorm;
    //////////////////////////////////////////////////////////////////////////
    //Test 
    int m_iTestPixelX;
    int m_iTestPixelY;

};


MED_IMAGING_END_NAMESPACE



#endif