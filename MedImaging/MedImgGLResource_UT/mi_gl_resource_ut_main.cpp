#include "gl/glew.h"
#include "gl/glut.h"
#include "mi_gl_environment.h"


using namespace MedImaging;

namespace
{
    void Display()
    {
        glClearColor(1.0,0.0,0.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glutSwapBuffers();
    }


    void Test()
    {
        GLEnvironment env;
        int iMajor(0) , iMinor(0);
        env.GetGLVersion(iMajor , iMinor);
        std::cout << "OpenGL Version : " << iMajor << "." << iMinor << std::endl;
        std::cout << "GL vendor : " << env.GetGLVendor() << std::endl;
        std::cout << "GL renderer : " << env.GetGLRenderer() << std::endl;
    }
}

void main(int argc , char* argv[])
{
    glutInit(&argc , argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(300,300);

    glutCreateWindow("Test GL resource");

    if ( GLEW_OK != glewInit())
    {
        std::cout <<"Init glew failed!\n";
        return;
    }

    glutDisplayFunc(Display);

    Test();

    glutMainLoop(); 
}
