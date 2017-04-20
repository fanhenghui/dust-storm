#include "gl/glew.h"

#include "MedImgCommon/mi_concurrency.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"
#include "MedImgGLResource/mi_gl_environment.h"
#include "MedImgGLResource/mi_gl_program.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"

#include "gl/freeglut.h"


using namespace MED_IMAGING_NAMESPACE;

namespace
{
    unsigned int m_uiTex;
    unsigned int m_uiProgram;
    unsigned int m_uiShader;
    GLProgram* m_pProgram;

    int m_iWidth = 800;
    int m_iHeight = 800;
    int m_iButton = -1;
    Point2 m_ptPre;

    void vglAttachShaderSource(GLuint prog, GLenum type, const char * source)
    {
        GLuint sh;

        sh = glCreateShader(type);
        glShaderSource(sh, 1, &source, NULL);
        glCompileShader(sh);
        char buffer[4096];
        glGetShaderInfoLog(sh, sizeof(buffer), NULL, buffer);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
    }

    #define STRINGIZE(a) #a
    static const char compute_shader_source[] =
        STRINGIZE(
#version 430 core\n

        layout (local_size_x = 4 , local_size_y = 4) in;

    layout (rgba32f , binding = 0) uniform image2D imgEntryPoint;

    void main()
    {
        imageStore(imgEntryPoint , ivec2(gl_GlobalInvocationID.xy) , vec4(1000,500,200, 100));
    }
    );

    //static const char compute_shader_source[] =
    //    "#version 430 core\n"
    //    "\n"
    //    "layout (local_size_x = 4, local_size_y = 4) in;\n"
    //    "\n"
    //    "layout (rgba32f) uniform image2D output_image;\n"
    //    "void main(void)\n"
    //    "{\n"
    //    "    imageStore(output_image,\n"
    //    "    ivec2(gl_GlobalInvocationID.xy),\n"
    //    "    vec4(vec2(gl_LocalInvocationID.xy) / vec2(gl_WorkGroupSize.xy), 0.0, 0.0));\n"
    //    "}\n"
    //    ;


    void Init()
    {
        CHECK_GL_ERROR;
        glGenTextures(1 , &m_uiTex);
        glBindTexture(GL_TEXTURE_2D , m_uiTex);
        //glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, m_iWidth, m_iHeight);
        /*RGBAUnit* pData = new RGBAUnit[m_iWidth*m_iHeight];
        for (int i = 0 ; i<m_iWidth*m_iHeight ; ++i)
        {
            pData[i].r = 255;
            pData[i].g = 0;
            pData[i].b = 0;
            pData[i].a = 255;
        }*/
        //glTexImage2D(GL_TEXTURE_2D , 0 , GL_RGBA8 , m_iWidth , m_iHeight , 0 , GL_RGBA , GL_UNSIGNED_BYTE , pData);
        GLTextureUtils::set_2d_wrap_s_t(GL_CLAMP_TO_BORDER);
        GLTextureUtils::set_filter(GL_TEXTURE_2D , GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D , 0 , GL_RGBA32F , m_iWidth , m_iHeight , 0 , GL_RGBA , GL_FLOAT, NULL);
        //glTexStorage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_iWidth, m_iHeight);
        //glBindTexture(GL_TEXTURE_2D , 0);

        //glDisable(GL_TEXTURE_2D);
        CHECK_GL_ERROR;

        m_uiProgram = glCreateProgram();
        vglAttachShaderSource(m_uiProgram, GL_COMPUTE_SHADER, compute_shader_source);
        glLinkProgram(m_uiProgram);

        //GLShaderInfo s(GL_COMPUTE_SHADER , ksMPREntryExitPointsComp , "");
        //unsigned int uiShader = glCreateShader(GL_COMPUTE_SHADER);
        //glShaderSource(uiShader , 1 , &ksMPREntryExitPointsComp , NULL);
        //glCompileShader(uiShader);
        //m_uiProgram = glCreateProgram();
        //glAttachShader(m_uiProgram , uiShader);
        //glLinkProgram(m_uiProgram);

        /*m_pProgram = new GLProgram(1);
        m_pProgram->set_shaders(std::vector<GLShaderInfo>(1,s));
        m_pProgram->initialize();
        m_pProgram->compile();*/
    }

    

    void Display()
    {
        glViewport(0,0,m_iWidth , m_iHeight);
        glClearColor(0.0,0.0,0.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        CHECK_GL_ERROR;

        glUseProgram(m_uiProgram);
        glBindImageTexture(0 , m_uiTex , 0 , GL_FALSE , 0 , GL_READ_WRITE , GL_RGBA32F);
        glDispatchCompute(200,200,1);
        glUseProgram(0);

        CHECK_GL_ERROR;

        glBindTexture(GL_TEXTURE_2D , m_uiTex);
        float *pData = new float[m_iWidth*m_iHeight*4];
        glGetTexImage(GL_TEXTURE_2D , 0 , GL_RGBA , GL_FLOAT , pData);
        glBindTexture(GL_TEXTURE_2D , 0);

        CHECK_GL_ERROR;

        //glDrawPixels(m_iWidth , m_iHeight , GL_RGBA , GL_UNSIGNED_BYTE , pData);

        delete [] pData;

        glutSwapBuffers();
    }

    void Keyboard(unsigned char key , int x , int y)
    {


        glutPostRedisplay();
    }

    void Resize(int x , int y)
    {
        m_iWidth = x;
        m_iHeight = y;
        glutPostRedisplay();
    }

    void Idle()
    {
        glutPostRedisplay();
    }

    void MouseClick(int button , int status , int x , int y)
    {
        m_iButton = button;

        if (m_iButton == GLUT_LEFT_BUTTON)
        {
            
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {

        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {

        }

        
        m_ptPre = Point2(x,y);
        glutPostRedisplay();
    }

    void MouseMotion(int x , int y)
    {
        Point2 ptCur(x,y);

        if (m_iButton == GLUT_LEFT_BUTTON)
        {

            
        }
        else if (m_iButton == GLUT_MIDDLE_BUTTON)
        {

        }
        else if (m_iButton == GLUT_RIGHT_BUTTON)
        {

        }

        m_ptPre = ptCur;
        glutPostRedisplay();

    }
}

void UT_CompureShader(int argc , char* argv[])
{
    try
    {
        glutInit(&argc , argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(0,0);
        glutInitWindowSize(m_iWidth,m_iHeight);

        glutCreateWindow("Test GL resource");

        if ( GLEW_OK != glewInit())
        {
            std::cout <<"Init glew failed!\n";
            return;
        }

        GLEnvironment env;
        int iMajor , iMinor;
        env.get_gl_version(iMajor , iMinor);

        glutDisplayFunc(Display);
        glutReshapeFunc(Resize);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);
        glutMouseFunc(MouseClick);
        glutMotionFunc(MouseMotion);

        Init();

        glutMainLoop(); 
    }
    catch (const Exception& e)
    {
        std::cout  << e.what();
        return;
    }
}