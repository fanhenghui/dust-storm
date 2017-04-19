#ifndef MED_IMAGING_ENTRY_EXIT_POINTS_H_
#define MED_IMAGING_ENTRY_EXIT_POINTS_H_

#include "MedImgRenderAlgorithm/mi_entry_exit_points.h"
#include "MedImgCommon/mi_common_define.h"
#include "MedImgArithmetic/mi_vector3f.h"
#include "MedImgArithmetic/mi_vector4f.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export MPREntryExitPoints : public EntryExitPoints
{
public:
    MPREntryExitPoints();

    virtual ~MPREntryExitPoints();

    virtual void Initialize();

    virtual void Finialize();

    void SetThickness(float fThickness);// In volume coordinate

    void SetSampleRate(float fSampleRate);

    virtual void CalculateEntryExitPoints();

    //////////////////////////////////////////////////////////////////////////
    /// \ plane function :  ax + by + cz = d
    /// \ Or normal&point x*N = d , thus d is the distance between original(0,0,0) and the plane
    /// \ vector4f (a,b,c,d)
    ///\ Entry plane's normal is form entry to exit
    ///\ Exit plane's normal is form exit to entry
    /// \In volume coordinate
    //////////////////////////////////////////////////////////////////////////
    void GetEntryExitPlane(Vector4f& vEntry , Vector4f& vExit , Vector3f& vRayDirNorm);

private:
    void CalEEPoints_CPU_i();

    void CalEEPlane_CPU_i();

    void CalEEPoints_GPU_i();

private:
    
    float m_fThickness;
    float m_fSampleRate;

    //Entry exit plane(orthogonal)
    Vector4f m_vEntryPlane;
    Vector4f m_vExitPlane;
    Vector3f m_vRayDirNorm;

    float m_fStandardSteps;

    //GPU entry exit points cal
    GLProgramPtr m_pProgram;

    //
    unsigned int m_uiTextTex;
};


MED_IMAGING_END_NAMESPACE

#endif