#include "gl/glew.h"
#include "gl/glut.h"
#include "mi_gl_environment.h"


using namespace medical_imaging;

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
        int major(0) , minor(0);
        env.get_gl_version(major , minor);
        std::cout << "OpenGL Version : " << major << "." << minor << std::endl;
        std::cout << "GL vendor : " << env.get_gl_vendor() << std::endl;
        std::cout << "GL renderer : " << env.get_gl_renderer() << std::endl;
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
