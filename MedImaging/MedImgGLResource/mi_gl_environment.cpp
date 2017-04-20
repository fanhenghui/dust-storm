#include "mi_gl_environment.h"
#include "gl/glew.h"

#include "MedImgCommon/mi_string_number_converter.h"

MED_IMAGING_BEGIN_NAMESPACE

GLEnvironment::GLEnvironment()
{

}

GLEnvironment::~GLEnvironment()
{

}

void GLEnvironment::get_gl_version(int &iMajor , int &iMinor)
{
    std::stringstream ss;
    ss << glGetString(GL_VERSION);
    std::string sVersionInfo = ss.str();

    StrNumConverter<int> strNumConv;

    std::string sMajor;
    sMajor.push_back(sVersionInfo[0]);
    std::string sMinor;
    sMinor.push_back(sVersionInfo[2]);

    iMajor = strNumConv.to_num(sMajor);
    iMinor = strNumConv.to_num(sMinor);
}

std::string GLEnvironment::get_gl_vendor()
{
    std::stringstream ss;
    ss << glGetString(GL_VENDOR);
    std::string sVendor = ss.str();
    return ss.str();
}

std::string GLEnvironment::get_gl_renderer()
{
    std::stringstream ss;
    ss << glGetString(GL_RENDERER);
    return ss.str();
}

MED_IMAGING_END_NAMESPACE