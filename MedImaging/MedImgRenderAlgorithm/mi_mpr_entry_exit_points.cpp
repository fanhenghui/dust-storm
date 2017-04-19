#include "mi_mpr_entry_exit_points.h"

#include <limits>
#include <time.h>
#include "boost/thread.hpp"

#include "MedImgCommon/mi_concurrency.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgArithmetic/mi_camera_base.h"
#include "MedImgArithmetic/mi_point2.h"
#include "MedImgArithmetic/mi_point3.h"
#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_arithmetic_utils.h"

#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "mi_shader_collection.h"
#include "mi_camera_calculator.h"

MED_IMAGING_BEGIN_NAMESPACE

namespace
{
    //Return true if out
    bool CheckOutside(Vector3f pt, Vector3f bound)
    {
        if (pt._m[0] <0 || pt._m[1] < 0 || pt._m[2] < 0 ||
            pt._m[0] > bound._m[0] || pt._m[1] > bound._m[1] || pt._m[2] > bound._m[2])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    //If ray[i] < FLOAT_EPSLION then set ray[i] = 1 adjust[i] = std::numeric_limits<float>::max()*0.5f
    bool RayIntersectAABBAcc(Vector3f ptRayStart, Vector3f ptMin, Vector3f vBound, Vector3f vRay, Vector3f vAdjust,
        float& fEntryStep, float& fExitStep)
    {
        Vector3f vBottomStep =  (ptMin - ptRayStart)/vRay;
        Vector3f vTopStep =  (ptMin + vBound - ptRayStart)/vRay;
        Vector3f vBottomStep2(vBottomStep);
        Vector3f vTopStep2(vTopStep);
        vBottomStep -= vAdjust;
        vTopStep -= vAdjust;
        vBottomStep2 += vAdjust;
        vTopStep2 += vAdjust;

        fEntryStep = vBottomStep.MinPerElem(vTopStep).MaxElem();
        fExitStep = vBottomStep2.MaxPerElem(vTopStep2).MinElem();

        //////////////////////////////////////////////////////////////////////////
        //fNear > fFar not intersected
        //fNear >0  fFar > 0 fNear <= fFar intersected , start point not arrive AABB yet
        //fNear <0 fFar > 0 intersected , start point is in AABB
        //fNear <0 fFar < 0 fNear < fFar , intersected , but start point is outside AABB in extension ray 
        return fEntryStep < fExitStep;
    }
}

MPREntryExitPoints::MPREntryExitPoints():m_fThickness(1.0f),m_vEntryPlane(1,0,0,0),m_vExitPlane(1,0,0,0),m_fSampleRate(1.0),m_uiTextTex(0)
{

}

MPREntryExitPoints::~MPREntryExitPoints()
{

}

void MPREntryExitPoints::SetSampleRate(float fSampleRate)
{
    m_fSampleRate = fSampleRate;
}

void MPREntryExitPoints::SetThickness(float fThickness)
{
    m_fThickness = fThickness;
}

void MPREntryExitPoints::CalculateEntryExitPoints()
{
    m_fStandardSteps = float(int(m_fThickness / m_fSampleRate + 0.5f));

    //clock_t t0 = clock();
    if (CPU_BASE == m_eStrategy)
    {
        CalEEPoints_CPU_i();
    }
    else if (CPU_BRICK_ACCELERATE == m_eStrategy)
    {
        CalEEPoints_CPU_i();
        CalEEPlane_CPU_i();
    }
    else if (GPU_BASE == m_eStrategy)
    {
        CalEEPoints_GPU_i();
    }
    //clock_t t1 = clock();
    //std::cout << "Calculate entry exit points cost : " << double(t1 - t0) << std::endl;
}

void MPREntryExitPoints::CalEEPoints_CPU_i()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pCamera);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pCameraCalculator);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pImgData);

        const Vector3f vDim((float)m_pImgData->m_uiDim[0],
            (float)m_pImgData->m_uiDim[1],
            (float)m_pImgData->m_uiDim[2]);

        //Calculate base plane of MPR
        const Matrix4 matV2W = m_pCameraCalculator->GetVolumeToWorldMatrix();
        const Matrix4 matVP = m_pCamera->GetViewProjectionMatrix();
        const Matrix4 matMVP = matVP*matV2W;
        const Matrix4 matMVPInv = matMVP.GetInverse();

        const Point3 pt00 = matMVPInv.Transform(Point3(-1.0,-1.0,0));
        const Point3 pt01 = matMVPInv.Transform(Point3(-1.0,1.0,0));
        const Point3 pt10 = matMVPInv.Transform(Point3(1.0,-1.0,0));
        const Vector3 vXDelta = (pt10 - pt00) * (1.0/(m_iWidth-1));
        const Vector3 vYDelta = (pt01 - pt00) * (1.0/(m_iHeight-1));

        Vector3 vViewDir = m_pCamera->GetLookAt() - m_pCamera->GetEye();
        vViewDir = matV2W.GetTranspose().Transform(vViewDir);
        vViewDir.Normalize();

        const Vector3f vRayDir = ArithmeticUtils::ConvertVector(vViewDir);

        const Vector3f pt00F((float)pt00.x , (float)pt00.y , (float)pt00.z);
        const Vector3f vXDeltaF((float)vXDelta.x,(float)vXDelta.y,(float)vXDelta.z);
        const Vector3f vYDeltaF((float)vYDelta.x,(float)vYDelta.y,(float)vYDelta.z);

        const float fThickness = m_fThickness;
        const float fThicknessHalf = fThickness*0.5f;
        Vector4f* pEntryPoints = m_pEntryBuffer.get();
        Vector4f* pExitPoints = m_pExitBuffer.get();

        //////////////////////////////////////////////////////////////////////////
        //Adjust ray direction
        Vector3f vRayBrick(vRayDir);
        Vector3f vRayBrickAdjust(0,0,0);
        for (int i = 0 ; i< 3 ; ++i)
        {
            if (fabs(vRayBrick._m[i]) <= FLOAT_EPSILON)
            {
                vRayBrick._m[i] = 1;//be divided
                vRayBrickAdjust._m[i] = std::numeric_limits<float>::max()*0.5f;
            }
        }
        //////////////////////////////////////////////////////////////////////////

        const int iTotalPixelNum = m_iWidth*m_iHeight;
#ifndef _DEBUG
#pragma omp parallel for
#endif
        for (int idx = 0 ; idx< iTotalPixelNum; ++idx)
        {
            Vector3f ptCurF;
            Vector3f ptEntryF;
            Vector3f ptExitF;
            Vector3f ptEntryIntersection;
            Vector3f ptExitIntersection;

            int iY = idx / m_iWidth;
            int iX = idx - iY*m_iWidth;

            ptCurF = pt00F + vXDeltaF*(float)iX + vYDeltaF*(float)iY;
            ptEntryF = ptCurF - vRayDir*fThicknessHalf;
            ptExitF = ptCurF + vRayDir*fThicknessHalf;

            ptEntryIntersection = ptEntryF;
            ptExitIntersection = ptExitF;

            //Intersect volume AABB to get intersected entry&exit points
            float fEntryStep(0), fExitStep(0);
            const bool bIntersection = RayIntersectAABBAcc(ptEntryF, Vector3f(0,0,0), vDim, vRayBrick, vRayBrickAdjust , fEntryStep, fExitStep);

            //Entry point outside
            if( CheckOutside(ptEntryF, vDim) )
            {
                if(!bIntersection || fEntryStep < 0 || fEntryStep > fThickness ) // check entry points in range of thickness and volume
                {
                    pEntryPoints[idx] = Vector4f(0,0,0,-1.0f);
                    pExitPoints[idx] = Vector4f(0,0,0,-1.0f);
                    continue;
                }
                ptEntryIntersection = ptEntryF + vRayDir*fEntryStep;
            }

            //Exit point outside
            if( CheckOutside(ptExitF, vDim) )
            {
                if(!bIntersection)
                {
                    pEntryPoints[idx] = Vector4f(0,0,0,-1.0f);
                    pExitPoints[idx] = Vector4f(0,0,0,-1.0f);
                    continue;
                }
                ptExitIntersection= ptEntryF + vRayDir*fExitStep;
            }

            //////////////////////////////////////////////////////////////////////////
            //alpha value : ray step
            float fStep = (float)(int)( (ptExitIntersection - ptEntryIntersection).Magnitude()/m_fSampleRate + 0.5f);
            if (fStep > m_fStandardSteps)//Adjust step to prevent  fStep = standard step + epsilon which it's ceil equals ( standard cell + 1)
            {
                fStep = m_fStandardSteps;
            }
            pEntryPoints[idx] = Vector4f(ptEntryIntersection,0.0f);//Entry step is 0 , the first sample position is on entry plane
            pExitPoints[idx] = Vector4f(ptExitIntersection,fStep);//Exit step is integer step which represent the integeration path

            //////////////////////////////////////////////////////////////////////////
        }

        /*Initialize();
        m_pEntryTex->Bind();
        m_pEntryTex->Load(GL_RGBA32F , m_iWidth , m_iHeight , GL_RGBA , GL_FLOAT , m_pEntryBuffer.get());

        m_pExitTex->Bind();
        m_pExitTex->Load(GL_RGBA32F , m_iWidth , m_iHeight , GL_RGBA , GL_FLOAT , m_pExitBuffer.get());

        m_pEntryTex->Bind();
        m_pEntryTex->Download(GL_RGBA , GL_FLOAT , m_pEntryBuffer.get());

        m_pExitTex->Bind();
        m_pExitTex->Download(GL_RGBA , GL_FLOAT , m_pExitBuffer.get());*/
    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::CalEEPlane_CPU_i()
{
    try
    {
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pCamera);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pCameraCalculator);
        RENDERALGO_CHECK_NULL_EXCEPTION(m_pImgData);

        Vector3f vDim((float)m_pImgData->m_uiDim[0],
            (float)m_pImgData->m_uiDim[1],
            (float)m_pImgData->m_uiDim[2]);

        //Calculate base plane of MPR
        const Matrix4 matV2W = m_pCameraCalculator->GetVolumeToWorldMatrix();
        const Matrix4 matVP = m_pCamera->GetViewProjectionMatrix();
        const Matrix4 matMVP = matVP*matV2W;
        const Matrix4 matMVPInv = matMVP.GetInverse();

        Vector3 vViewDir = m_pCamera->GetLookAt() - m_pCamera->GetEye();
        vViewDir = matV2W.GetTranspose().Transform(vViewDir);
        vViewDir.Normalize();
        const Vector3 vRayDir = vViewDir;
        m_vRayDirNorm = ArithmeticUtils::ConvertVector(vRayDir);

        const float fThickness = m_fThickness;
        const float fThicknessHalf = fThickness*0.5f;

        const Point3 ptCenter = matMVPInv.Transform(Point3(0.0,0.0,0));
        const Point3 ptEntry = ptCenter - vRayDir*fThicknessHalf;
        const Point3 ptExit = ptCenter + vRayDir*fThicknessHalf;

        double dDisEntry = vRayDir.DotProduct(ptEntry - Point3::kZeroPoint);
        double dDisExit = (-vRayDir).DotProduct(ptExit - Point3::kZeroPoint);

        m_vEntryPlane = Vector4f((float)vRayDir.x ,(float)vRayDir.y , (float)vRayDir.z , (float)dDisEntry);
        m_vExitPlane = Vector4f(-(float)vRayDir.x ,-(float)vRayDir.y , -(float)vRayDir.z , (float)dDisExit);

    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::GetEntryExitPlane(Vector4f& vEntry , Vector4f& vExit , Vector3f& vRayDirNorm)
{
    vEntry = m_vEntryPlane;
    vExit = m_vExitPlane;
    vRayDirNorm = m_vRayDirNorm;
}

void MPREntryExitPoints::CalEEPoints_GPU_i()
{
    try
    {
#define IMAGE_ENTRY_POINT 0
#define IMAGE_EXIT_POINT 1
#define DISPLAY_SIZE 2
#define VOLUME_DIM 3
#define MVP_INVERSE 4
#define THICKNESS 5
#define RAY_DIRECTION 6

        CHECK_GL_ERROR;

        Initialize();

        const unsigned int uiProgram = m_pProgram->GetID();
        if (0 ==uiProgram)
        {
            RENDERALGO_THROW_EXCEPTION("Program ID is 0!");
        }

        m_pProgram->Bind();

        glPushAttrib(GL_ALL_ATTRIB_BITS);


        m_pEntryTex->BindImage(IMAGE_ENTRY_POINT , 0 , false , 0 , GL_READ_WRITE , GL_RGBA32F);
        m_pExitTex->BindImage(IMAGE_EXIT_POINT , 0 , false , 0 , GL_READ_WRITE , GL_RGBA32F);

        CHECK_GL_ERROR;

        glProgramUniform2ui(uiProgram , DISPLAY_SIZE , (GLuint)m_iWidth , (GLuint)m_iHeight);

        CHECK_GL_ERROR;

        const float fDim[3] = {(float)m_pImgData->m_uiDim[0] , (float)m_pImgData->m_uiDim[1] , (float)m_pImgData->m_uiDim[2]};
        glProgramUniform3f(uiProgram , VOLUME_DIM , fDim[0] , fDim[1] , fDim[2]);

        CHECK_GL_ERROR;

        const Matrix4 matV2W = m_pCameraCalculator->GetVolumeToWorldMatrix();
        const Matrix4 matVP = m_pCamera->GetViewProjectionMatrix();
        const Matrix4 matMVP = matVP*matV2W;
        const Matrix4 matMVPInv = matMVP.GetInverse();

        CHECK_GL_ERROR;

        float fMat[16] = {0.0f};
        matMVPInv.ToFloat16(fMat);
        glProgramUniformMatrix4fv(uiProgram , MVP_INVERSE , 1 , GL_FALSE , fMat);

        CHECK_GL_ERROR;

        glProgramUniform1f(uiProgram , THICKNESS , m_fThickness);

        CHECK_GL_ERROR;

        Vector3 vViewDir = m_pCamera->GetLookAt() - m_pCamera->GetEye();
        vViewDir = matV2W.GetTranspose().Transform(vViewDir);
        vViewDir.Normalize();
        glProgramUniform3f(uiProgram , RAY_DIRECTION , (float)vViewDir.x ,(float)vViewDir.y , (float)vViewDir.z);

        CHECK_GL_ERROR;

        const unsigned int aLocalWorkGroupCount[2] = {4,4};
        unsigned int aWorkGroupsNum[2] = {(unsigned int)m_iWidth/aLocalWorkGroupCount[0], (unsigned int)m_iHeight/aLocalWorkGroupCount[1]};
        if (aWorkGroupsNum[0]*aLocalWorkGroupCount[0] != (unsigned int)m_iWidth)
        {
            aWorkGroupsNum[0] +=1;
        }
        if (aWorkGroupsNum[1]*aLocalWorkGroupCount[1] != (unsigned int)m_iHeight)
        {
            aWorkGroupsNum[1] +=1;
        }
        glDispatchCompute(aWorkGroupsNum[0] , aWorkGroupsNum[1] , 1);

        glPopAttrib();

        m_pProgram->UnBind();

        CHECK_GL_ERROR;

        //////////////////////////////////////////////////////////////////////////
        //For testing
        /*std::cout << "Size : " << m_iWidth << " " << m_iHeight << std::endl;
        m_pEntryTex->Bind();
        m_pEntryTex->Download(GL_RGBA , GL_FLOAT , m_pEntryBuffer.get());*/

        //m_pExitTex->Bind();
        //m_pExitTex->Download(GL_RGBA , GL_FLOAT , m_pExitBuffer.get());

        //this->DebugOutputEntryPoints("D:/temp/gpu_MPR_ee.raw");

        CHECK_GL_ERROR;

#undef IMAGE_ENTRY_POINT
#undef IMAGE_EXIT_POINT
#undef DISPLAY_SIZE
#undef VOLUME_DIM
#undef MVP_INVERSE
#undef THICKNESS
#undef RAY_DIRECTION

    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        assert(false);
        throw e;
    }
}

void MPREntryExitPoints::Initialize()
{
    EntryExitPoints::Initialize();

    if (GPU_BASE == m_eStrategy)
    {
        if (!m_pProgram)
        {
            UIDType uid = 0;
            m_pProgram = GLResourceManagerContainer::Instance()->GetProgramManager()->CreateObject(uid);
            m_pProgram->SetDescription("MPR entry exit program");
            m_pProgram->SetShaders(std::vector<GLShaderInfo>(1 , GLShaderInfo(GL_COMPUTE_SHADER , ksMPREntryExitPointsComp , "MPR entry exit compute shader")));
            m_pProgram->Initialize();
            m_pProgram->Compile();
        }
    }
}

void MPREntryExitPoints::Finialize()
{
    EntryExitPoints::Finialize();

    if (m_pProgram)
    {
        GLResourceManagerContainer::Instance()->GetProgramManager()->RemoveObject(m_pProgram->GetUID());
        m_pProgram.reset();
        GLResourceManagerContainer::Instance()->GetProgramManager()->Update();
    }
}





MED_IMAGING_END_NAMESPACE