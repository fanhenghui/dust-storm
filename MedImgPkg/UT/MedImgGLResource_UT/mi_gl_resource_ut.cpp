#include "GL/glew.h"
#include "GL/glxew.h"

#include "MedImgGLResource/mi_gl_context.h"

using namespace medical_imaging;

int main(int argc , char* argv[])
{
    GLContext context(100);
    context.initialize();
    context.make_current();
    context.finalize();

    return 0;
}