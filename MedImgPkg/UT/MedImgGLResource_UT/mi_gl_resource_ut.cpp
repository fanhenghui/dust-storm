#include "GL/glew.h"
#include "GL/glxew.h"

#include "MedImgGLResource/mi_gl_context.h"
#include "MedImgGLResource/mi_gl_utils.h"

#include <unistd.h>
#include "GL/glew.h"

#include "boost/thread.hpp"

using namespace medical_imaging;

// GLContext context(100);


// void process_operation()
// {
//     context.make_current(2);
//     for(;;){
        
//         sleep(1);
//         std::cout << "Operation : ";
//         const GLubyte *s = glGetString(GL_VERSION);
//         printf("GL Version = %s\n", s);   
//     }
// }

// void process_render()
// {
//     context.make_current(1);
//     for(;;){
//         CHECK_GL_ERROR;
//         GLuint _fbo_id = 0;
//         glGenFramebuffers(1 , &_fbo_id);
//         //glDeleteFramebuffers(1 , &_fbo_id);
//         CHECK_GL_ERROR;
//         sleep(2);
//         std::cout << "Rendering : " << _fbo_id;
//         const GLubyte *s = glGetString(GL_VERSION);
//         printf("GL Version = %s\n", s);   
//     }
// }

// int main(int argc , char* argv[])
// {    
//     context.initialize();
//     context.create_shared_context(1);
//     context.create_shared_context(2);
//     context.make_current();

//     CHECK_GL_ERROR;

//     GLuint id = 0;
//     GLuint ids[2];
//     //glGenTextures(2 ,ids );
//     glGenVertexArrays(2 , ids);


//     CHECK_GL_ERROR;

//     glPushAttrib(GL_ALL_ATTRIB_BITS);


//     CHECK_GL_ERROR;

//     glPopAttrib();

//     CHECK_GL_ERROR;

//     std::cout << "DONE\n";
//     return 0;


//     boost::thread th(process_operation);
//     boost::thread th1(process_render);

//     std::string s;
//     while(std::cin >> s)
//     {
//         std::cout << "Main : " << s << "\t:";
//         const GLubyte *s = glGetString(GL_VERSION);
//         printf("GL Version = %s\n", s);   

        
//     }

//     th.join();
//     th1.join();

//     context.finalize();

//     return 0;
// }


#include "GL/freeglut.h"

static void display()
{
    glViewport(0,0,800,600);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glBegin(GL_TRIANGLES);
    glVertex2f(-0.5,-0.5);
    glVertex2f(0.5,-0.5);
    glVertex2f(0.5,0.5);
    glEnd();
    glPopAttrib();


    glutSwapBuffers();
}


int main(int argc , char *argv[])
{
    glutInit(&argc , argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(800, 600);
	glutCreateWindow("test gl resource");
	
	GLenum err = glewInit();
	if (GLEW_OK != err){
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		return 1;
	}
	
	glutDisplayFunc(display);
	glutMainLoop();

    return 0;
}