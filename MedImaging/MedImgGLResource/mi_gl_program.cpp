#include "mi_gl_program.h"
#include "mi_gl_utils.h"

MED_IMAGING_BEGIN_NAMESPACE

    namespace
{
    std::string GetShaderTypeString(GLuint uiType)
    {
        switch (uiType)
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

GLProgram::GLProgram(UIDType uid):GLObject(uid),m_uiProgramID(0)
{
    set_type("GLProgram");
}

GLProgram::~GLProgram()
{

}

void GLProgram::initialize()
{
    if (0 == m_uiProgramID)
    {
        m_uiProgramID = glCreateProgram();
    }
}

void GLProgram::finalize()
{
    if (0 != m_uiProgramID)
    {
        glDeleteProgram(m_uiProgramID);
        m_uiProgramID = 0;
    }
}

unsigned int GLProgram::get_id() const
{
    return m_uiProgramID;
}

void GLProgram::compile()
{
    if (m_Shaders.empty())
    {
        GLRESOURCE_THROW_EXCEPTION("Shaders is empty!");
    }

    initialize();

    for (auto it = m_Shaders.begin(); it != m_Shaders.end(); ++it)
    {
        GLuint uiShader = glCreateShader((*it).m_eType);
        (*it).m_uiShaderID = uiShader;
        glShaderSource(uiShader, 1, &((*it).m_ksContext), NULL);

        glCompileShader(uiShader);

        //ºÏ≤È±‡“Î◊¥Ã¨
        GLint iComplied(-1);
        glGetShaderiv(uiShader, GL_COMPILE_STATUS, &iComplied);
        if (!iComplied)
        {
            //ªÒ»°±‡“Î¥ÌŒÛ–≈œ¢µƒ≥§∂»
            GLsizei uiLen(0);
            glGetShaderiv(uiShader, GL_INFO_LOG_LENGTH, &uiLen);

            //ªÒ»°±‡“Î¥ÌŒÛ–≈œ¢
            GLchar *pLogInfo = new GLchar[uiLen + 1];
            glGetShaderInfoLog(uiShader, uiLen, &uiLen, pLogInfo);
            pLogInfo[uiLen] = '\0';

            //…æ≥˝ Shader “‘∑¿÷πœ‘¥Ê–π¬∂
            for (auto itDelete = m_Shaders.begin(); itDelete != m_Shaders.end(); ++itDelete)
            {
                if (0 != (*itDelete).m_uiShaderID)
                {
                    glDeleteShader((*itDelete).m_uiShaderID);
                    (*itDelete).m_uiShaderID = 0;
                }
            }

            //¥Ú”°±‡“Î¥ÌŒÛ–≈œ¢ 
            std::string sLogInfo = pLogInfo;
            delete[] pLogInfo;

            std::string strErrorInfo = GetShaderTypeString((*it).m_eType) + std::string(" \"") + (*it).m_sShaderName + std::string("\" compiled failed : ") + sLogInfo + std::string("\n");
            std::cout << strErrorInfo;
            GLRESOURCE_THROW_EXCEPTION(strErrorInfo);
        }

        glAttachShader(m_uiProgramID, uiShader);
    }

    glLinkProgram(m_uiProgramID);

    GLint iLinked(-1);
    glGetProgramiv(m_uiProgramID, GL_LINK_STATUS, &iLinked);
    if (!iLinked)
    {
        GLsizei uiLen(0);
        glGetProgramiv(m_uiProgramID, GL_INFO_LOG_LENGTH, &uiLen);

        GLchar* pLogInfo = new GLchar[uiLen + 1];
        glGetProgramInfoLog(m_uiProgramID, uiLen, &uiLen, pLogInfo);
        pLogInfo[uiLen] = '\0';

        std::string sLogInfo = pLogInfo;
        delete[] pLogInfo;

        //…æ≥˝Shader “‘º∞ Program ∑¿÷πœ‘¥Ê–π¬∂
        for (auto itDelete = m_Shaders.begin(); itDelete != m_Shaders.end(); ++itDelete)
        {
            glDeleteShader((*itDelete).m_uiShaderID);
            (*itDelete).m_uiShaderID = 0;
        }

        glDeleteProgram(m_uiProgramID);

        //¥Ú”°¡¥Ω”¥ÌŒÛ–≈œ¢
        std::string strErrorInfo = std::string("Program link failed : ") + sLogInfo + std::string("\n");
        std::cout << strErrorInfo;
        COMMON_THROW_EXCEPTION(strErrorInfo.c_str());

    }

    for (auto it = m_Shaders.begin(); it != m_Shaders.end(); ++it)
    {
        glDeleteShader((*it).m_uiShaderID);
        (*it).m_uiShaderID = 0;
    }

    CHECK_GL_ERROR;
}

void GLProgram::set_shaders(std::vector<GLShaderInfo> shaders)
{
    m_Shaders = shaders;
}

void GLProgram::bind()
{
    glUseProgram(m_uiProgramID);
}

void GLProgram::unbind()
{
    glUseProgram(0);
}

int GLProgram::get_uniform_location(const char* sName)
{
    return glGetUniformLocation(m_uiProgramID, sName);
}

MED_IMAGING_END_NAMESPACE



