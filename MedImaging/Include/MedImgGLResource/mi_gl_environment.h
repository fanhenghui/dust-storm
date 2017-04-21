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
    void get_gl_version(int &major , int &minor);

    //OpenGL 的实现厂商
    std::string get_gl_vendor();

    //OpenGL 渲染器（硬件平台）
    std::string get_gl_renderer();

protected:

private:
};

MED_IMAGING_END_NAMESPACE

#endif