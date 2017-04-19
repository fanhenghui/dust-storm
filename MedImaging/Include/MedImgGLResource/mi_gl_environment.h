#ifndef MED_IMAGING_GL_ENVIRONMENT_H_
#define MED_IMAGING_GL_ENVIRONMENT_H_

#include "MedImgGLResource/mi_gl_resource_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLEnvironment
{
public:
    GLEnvironment();

    ~GLEnvironment();

    //OpenGL 版本信息
    void GetGLVersion(int &iMajor , int &iMinor);

    //OpenGL 的实现厂商
    std::string GetGLVendor();

    //OpenGL 渲染器（硬件平台）
    std::string GetGLRenderer();

protected:

private:
};

MED_IMAGING_END_NAMESPACE

#endif