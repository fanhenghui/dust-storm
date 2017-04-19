#ifndef MED_IMAGING_RAY_CASTER_CANVAS_H_
#define MED_IMAGING_RAY_CASTER_CANVAS_H_

#include "MedImgRenderAlgorithm/mi_render_algo_stdafx.h"
#include "MedImgGLResource/mi_gl_resource_define.h"
#include "MedImgArithmetic/mi_color_unit.h"
#include "MedImgCommon/mi_common_define.h"

MED_IMAGING_BEGIN_NAMESPACE

class RenderAlgo_Export RayCasterCanvas
{
public:
    RayCasterCanvas();

    ~RayCasterCanvas();

    void Initialize();

    void Finialize();

    //void SetDataType(DataType eDataType);

    void SetDisplaySize(int iWidth , int iHeight);

    void GetDisplaySize(int& iWidth, int& iHeight) const;

    GLFBOPtr GetFBO();

    GLTexture2DPtr GetColorAttachTexture();

    GLTexture2DPtr GetGrayAttachTexture();

    RGBAUnit* GetColorArray();

    //If change data type or display size. Should call it to update FBO and mapped array
    void UpdateFBO();

    void UploadColorArray();

    //void* GetGrayArray();

    //void DownloadGrayArray();

public:
    void DebugOutputColor(const std::string& sFileName);

    //void DebugOutputGray(const std::string& sFileName);

protected:

private:
    GLFBOPtr m_pFBO;
    GLTexture2DPtr m_pColorAttach0;//For RGBA Color
    //GLTexture2DPtr m_pGrayAttach1;//For Raw //TODO if not save image to gray 16bit ,then its useless
    GLTexture2DPtr m_pDepthAttach;
    int m_iWidth;
    int m_iHeight;
    //DataType m_eDataType;
    std::unique_ptr<RGBAUnit[]> m_pColorArray;
    //std::unique_ptr<char[]> m_pGrayArray;
    bool m_bInit;
};

MED_IMAGING_END_NAMESPACE


#endif