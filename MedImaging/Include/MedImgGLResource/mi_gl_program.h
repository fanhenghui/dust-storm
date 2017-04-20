#ifndef MED_IMAGING_GL_PROGRAM_H_
#define MED_IMAGING_GL_PROGRAM_H_

#include "mi_gl_object.h"

MED_IMAGING_BEGIN_NAMESPACE

typedef struct _ShaderInfo
{
    GLenum m_eType;
    const char* m_ksContext;
    std::string m_sShaderName;
    GLuint m_uiShaderID;
    _ShaderInfo(GLenum shaderType, const char* shaderContext, const std::string &sShaderName)
        :m_uiShaderID(0), m_eType(shaderType), m_ksContext(shaderContext), m_sShaderName(sShaderName)
    {}
}GLShaderInfo;

class GLResource_Export GLProgram : public GLObject
{
public:
    GLProgram(UIDType uid);

    ~GLProgram();

    virtual void initialize();

    virtual void finalize();

    unsigned int get_id() const;

    void compile();

    void set_shaders(std::vector<GLShaderInfo> shaders);

    void bind();

    void unbind();

    int get_uniform_location(const char* sName);

private:
    std::vector<GLShaderInfo> m_Shaders;
    unsigned int m_uiProgramID;
};

MED_IMAGING_END_NAMESPACE

#endif

