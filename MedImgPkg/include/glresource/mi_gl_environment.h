#ifndef MED_IMG_GL_ENVIRONMENT_H_
#define MED_IMG_GL_ENVIRONMENT_H_

#include "glresource/mi_gl_resource_export.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLEnvironment
{
public:
    GLEnvironment();

    ~GLEnvironment();

    //OpenGL �汾��Ϣ
    void get_gl_version(int &major , int &minor);

    //OpenGL ��ʵ�ֳ���
    std::string get_gl_vendor();

    //OpenGL ��Ⱦ����Ӳ��ƽ̨��
    std::string get_gl_renderer();

protected:

private:
};

MED_IMG_END_NAMESPACE

#endif