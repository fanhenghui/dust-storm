#ifndef MEDIMGRESOURCE_GL_PROGRAM_H_
#define MEDIMGRESOURCE_GL_PROGRAM_H_

#include "mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE

struct GLShaderInfo {
    GLenum type;
    const char* context;
    std::string shader_name;
    GLuint shader_id;

    GLShaderInfo(GLenum shader_type, const char* shader_context,
                 const std::string& shader_name)
        : shader_id(0), type(shader_type), context(shader_context),
          shader_name(shader_name) {}
};

class GLResource_Export GLProgram : public GLObject {
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

    int get_uniform_location(const char* name);

private:
    std::vector<GLShaderInfo> _shaders;
    unsigned int _program_id;
};

MED_IMG_END_NAMESPACE

#endif
