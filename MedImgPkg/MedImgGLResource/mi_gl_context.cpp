#include "mi_gl_context.h"

#ifdef WIN32

#else

#include "GL/glew.h"
#include "GL/glxew.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#endif

MED_IMG_BEGIN_NAMESPACE

XGLContext::XGLContext(UIDType uid):GLObject(uid)
{

}

XGLContext::~XGLContext()
{

}

void XGLContext::initialize()
{

}

void XGLContext::finalize()
{

}

void XGLContext::make_current()
{

}

void XGLContext::make_noncurrent()
{
    
}

MED_IMG_END_NAMESPACE