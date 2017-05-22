#include "gl_program.h"
#include "gl_utils.h"



    namespace
{
    std::string GetShaderTypeString(GLenum type)
    {
        switch (type)
        {
        case GL_VERTEX_SHADER:
            {
                return "Vertex shader";
            }
        case GL_GEOMETRY_SHADER:
            {
                return "Geometry shader";
            }
        case GL_TESS_CONTROL_SHADER:
            {
                return "Tesselation control shader";
            }
        case GL_TESS_EVALUATION_SHADER:
            {
                return "Tesselation evaluation shader";
            }
        case GL_FRAGMENT_SHADER:
            {
                return "Fragment shader";
            }
        case GL_COMPUTE_SHADER:
            {
                return "Compute shader";
            }
        default:
            return "Undefied shader";
        }
    }
}

GLProgram::GLProgram(UIDType uid):GLObject(uid),_program_id(0)
{
    set_type("GLProgram");
}

GLProgram::~GLProgram()
{

}

void GLProgram::initialize()
{
    if (0 == _program_id)
    {
        _program_id = glCreateProgram();
    }
}

void GLProgram::finalize()
{
    if (0 != _program_id)
    {
        glDeleteProgram(_program_id);
        _program_id = 0;
    }
}

unsigned int GLProgram::get_id() const
{
    return _program_id;
}

void GLProgram::compile()
{
    if (_shaders.empty())
    {
        return;
        //GLRESOURCE_THROW_EXCEPTION("Shaders is empty!");
    }

    initialize();

    for (auto it = _shaders.begin(); it != _shaders.end(); ++it)
    {
        GLuint shader = glCreateShader((*it).type);
        (*it).shader_id = shader;
        glShaderSource(shader, 1, &((*it).context), NULL);

        glCompileShader(shader);

        //ºÏ≤È±‡“Î◊¥Ã¨
        GLint compile_status(-1);
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
        if (!compile_status)
        {
            //ªÒ»°±‡“Î¥ÌŒÛ–≈œ¢µƒ≥§∂»
            GLsizei log_length(0);
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

            //ªÒ»°±‡“Î¥ÌŒÛ–≈œ¢
            std::unique_ptr<GLchar[]> log_context(new GLchar[log_length + 1]);
            glGetShaderInfoLog(shader, log_length, &log_length, log_context.get());
            log_context[log_length] = '\0';

            //…æ≥˝ Shader “‘∑¿÷πœ‘¥Ê–π¬∂
            for (auto it_delete = _shaders.begin(); it_delete != _shaders.end(); ++it_delete)
            {
                if (0 != (*it_delete).shader_id)
                {
                    glDeleteShader((*it_delete).shader_id);
                    (*it_delete).shader_id = 0;
                }
            }

            //¥Ú”°±‡“Î¥ÌŒÛ–≈œ¢ 
            std::string err_info = GetShaderTypeString((*it).type) + std::string(" \"") + (*it).shader_name + std::string("\" compiled failed : ") + std::string(log_context.get()) + std::string("\n");
            std::cout << err_info;
            //GLRESOURCE_THROW_EXCEPTION(err_info);
        }

        glAttachShader(_program_id, shader);
    }

    glLinkProgram(_program_id);

    GLint link_status(-1);
    glGetProgramiv(_program_id, GL_LINK_STATUS, &link_status);
    if (!link_status)
    {
        GLsizei log_length(0);
        glGetProgramiv(_program_id, GL_INFO_LOG_LENGTH, &log_length);

        std::unique_ptr<GLchar[]> log_context(new GLchar[log_length + 1]);
        glGetProgramInfoLog(_program_id, log_length, &log_length, log_context.get());
        log_context[log_length] = '\0';

        //…æ≥˝Shader “‘º∞ Program ∑¿÷πœ‘¥Ê–π¬∂
        for (auto it_delete = _shaders.begin(); it_delete != _shaders.end(); ++it_delete)
        {
            glDeleteShader((*it_delete).shader_id);
            (*it_delete).shader_id = 0;
        }

        glDeleteProgram(_program_id);

        //¥Ú”°¡¥Ω”¥ÌŒÛ–≈œ¢
        std::string err_info = std::string("Program link failed : ") + std::string(log_context.get()) + std::string("\n");
        std::cout << err_info;
        //COMMON_THROW_EXCEPTION(err_info.c_str());
    }

    for (auto it = _shaders.begin(); it != _shaders.end(); ++it)
    {
        glDeleteShader((*it).shader_id);
        (*it).shader_id = 0;
    }

    CHECK_GL_ERROR;
}

void GLProgram::set_shaders(std::vector<GLShaderInfo> shaders)
{
    _shaders = shaders;
}

void GLProgram::bind()
{
    glUseProgram(_program_id);
}

void GLProgram::unbind()
{
    glUseProgram(0);
}

int GLProgram::get_uniform_location(const char* name)
{
    return glGetUniformLocation(_program_id, name);
}





