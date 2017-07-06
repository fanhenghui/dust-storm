#include "GL/glew.h"
#include "GL/glxew.h"

#include "MedImgGLResource/mi_gl_context.h"

#include <unistd.h>
#include "boost/thread.hpp"

using namespace medical_imaging;

GLContext context(100);

void process_operation()
{
    context.make_current(2);
    for(;;){

        sleep(1);
        std::cout << "Operation : ";
        const GLubyte *s = glGetString(GL_VERSION);
        printf("GL Version = %s\n", s);   
    }
}

void process_render()
{
    context.make_current(1);
    for(;;){

        sleep(2);
        std::cout << "Rendering : " ;
        const GLubyte *s = glGetString(GL_VERSION);
        printf("GL Version = %s\n", s);   
    }
}

int main(int argc , char* argv[])
{    
    context.initialize();
    context.create_shared_context(1);
    context.create_shared_context(2);
    context.make_current();

    boost::thread th(process_operation);
    boost::thread th1(process_render);

    std::string s;
    while(std::cin >> s)
    {
        std::cout << "Main : " << s << "\t:";
        const GLubyte *s = glGetString(GL_VERSION);
        printf("GL Version = %s\n", s);   
    }

    th.join();
    th1.join();

    context.finalize();

    return 0;
}