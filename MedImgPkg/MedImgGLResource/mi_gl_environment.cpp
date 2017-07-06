#include "mi_gl_environment.h"
#include "GL/glew.h"

#include "MedImgUtil/mi_string_number_converter.h"

MED_IMG_BEGIN_NAMESPACE

GLEnvironment::GLEnvironment()
{

}

GLEnvironment::~GLEnvironment()
{

}

void GLEnvironment::get_gl_version(int &major , int &minor)
{
    std::stringstream ss;
    ss << glGetString(GL_VERSION);
    std::string version_info = ss.str();

    StrNumConverter<int> num_converter;

    std::string s_major;
    s_major.push_back(version_info[0]);
    std::string s_minor;
    s_minor.push_back(version_info[2]);

    major = num_converter.to_num(s_major);
    minor = num_converter.to_num(s_minor);
}

std::string GLEnvironment::get_gl_vendor()
{
    std::stringstream ss;
    ss << glGetString(GL_VENDOR);
    std::string vendor = ss.str();
    return ss.str();
}

std::string GLEnvironment::get_gl_renderer()
{
    std::stringstream ss;
    ss << glGetString(GL_RENDERER);
    return ss.str();
}

MED_IMG_END_NAMESPACE