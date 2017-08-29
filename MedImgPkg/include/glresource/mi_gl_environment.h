#ifndef MEDIMGRESOURCE_GL_ENVIRONMENT_H_
#define MEDIMGRESOURCE_GL_ENVIRONMENT_H_

#include "glresource/mi_gl_resource_export.h"

MED_IMG_BEGIN_NAMESPACE

class GLResource_Export GLEnvironment {
public:
    GLEnvironment();

    ~GLEnvironment();

    void get_gl_version(int& major, int& minor);

    std::string get_gl_vendor();

    std::string get_gl_renderer();

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif