#ifndef MED_IMAGING_GL_ENVIRONMENT_H_
#define MED_IMAGING_GL_ENVIRONMENT_H_

#include "MedImgGLResource/mi_gl_resource_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

class GLResource_Export GLEnvironment
{
public:
    GLEnvironment();

    ~GLEnvironment();

    //OpenGL �汾��Ϣ
    void GetGLVersion(int &iMajor , int &iMinor);

    //OpenGL ��ʵ�ֳ���
    std::string GetGLVendor();

    //OpenGL ��Ⱦ����Ӳ��ƽ̨��
    std::string GetGLRenderer();

protected:

private:
};

MED_IMAGING_END_NAMESPACE

#endif