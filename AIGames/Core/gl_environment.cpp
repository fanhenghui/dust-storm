#include "gl_environment.h"
#include "Ext/gl/glew.h"

#include "common_utils.h"



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


    std::string s_major;
    s_major.push_back(version_info[0]);
    std::string s_minor;
    s_minor.push_back(version_info[2]);

    major = string_to_num<int>(s_major);
    minor = string_to_num<int>(s_minor);
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

