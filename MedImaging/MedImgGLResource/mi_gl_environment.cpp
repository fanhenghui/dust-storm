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

void GLEnvironment::GetGLVersion(int &iMajor , int &iMinor)
{
    std::stringstream ss;
    ss << glGetString(GL_VERSION);
    std::string sVersionInfo = ss.str();

    StrNumConverter<int> strNumConv;

    std::string sMajor;
    sMajor.push_back(sVersionInfo[0]);
    std::string sMinor;
    sMinor.push_back(sVersionInfo[2]);

    iMajor = strNumConv.ToNumber(sMajor);
    iMinor = strNumConv.ToNumber(sMinor);
}

std::string GLEnvironment::GetGLVendor()
{
    std::stringstream ss;
    ss << glGetString(GL_VENDOR);
    std::string sVendor = ss.str();
    return ss.str();
}

std::string GLEnvironment::GetGLRenderer()
{
    std::stringstream ss;
    ss << glGetString(GL_RENDERER);
    return ss.str();
}

MED_IMAGING_END_NAMESPACE