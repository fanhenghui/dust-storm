#include "mi_gl_context.h"

#ifdef WIN32

#else

#include <stdio.h>
#include <stdlib.h>

#endif

MED_IMG_BEGIN_NAMESPACE

XGLContext::XGLContext(UIDType uid):GLObject(uid),_dpy(nullptr),_ctx(NULL),_win((Window)NULL)
{

}

XGLContext::~XGLContext()
{

}

void XGLContext::early_init_glx_fn_pointers()
{

    glGenVertexArraysAPPLE = (void(*)(GLsizei, const GLuint*))glXGetProcAddressARB((GLubyte*)"glGenVertexArrays");
    glBindVertexArrayAPPLE = (void(*)(const GLuint))glXGetProcAddressARB((GLubyte*)"glBindVertexArray");
    glDeleteVertexArraysAPPLE = (void(*)(GLsizei, const GLuint*))glXGetProcAddressARB((GLubyte*)"glGenVertexArrays");
    glXCreateContextAttribsARB = (GLXContext(*)(Display* dpy, GLXFBConfig config, GLXContext share_context, Bool direct, const int *attrib_list))glXGetProcAddressARB((GLubyte*)"glXCreateContextAttribsARB");
    glXChooseFBConfig = (GLXFBConfig*(*)(Display *dpy, int screen, const int *attrib_list, int *nelements))glXGetProcAddressARB((GLubyte*)"glXChooseFBConfig");
    glXGetVisualFromFBConfig = (XVisualInfo*(*)(Display *dpy, GLXFBConfig config))glXGetProcAddressARB((GLubyte*)"glXGetVisualFromFBConfig");
}

void XGLContext::create_window()
{
    XSetWindowAttributes winAttribs;
    GLint winmask;
    GLint nMajorVer = 0;
    GLint nMinorVer = 0;
    XVisualInfo *visualInfo;
    GLXFBConfig *fbConfigs;
    int numConfigs = 0;
    static int fbAttribs[] = {
                    GLX_RENDER_TYPE,   GLX_RGBA_BIT,
                    GLX_X_RENDERABLE,  True,
                    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
                    GLX_DOUBLEBUFFER,  True,
                    GLX_RED_SIZE,      8,
                    GLX_BLUE_SIZE,     8,
                    GLX_GREEN_SIZE,    8,
                    0 };

    early_init_glx_fn_pointers();

    // Tell X we are going to use the display
    _dpy = XOpenDisplay(NULL);

    // Get Version info
    glXQueryVersion(_dpy, &nMajorVer, &nMinorVer);
    printf("Supported GLX version - %d.%d\n", nMajorVer, nMinorVer);   

    if(nMajorVer == 1 && nMinorVer < 2)
    {
        printf("ERROR: GLX 1.2 or greater is necessary\n");
        XCloseDisplay(_dpy);
        exit(0);
    }
    // Get a new fb config that meets our attrib requirements
    fbConfigs = glXChooseFBConfig(_dpy, DefaultScreen(_dpy), fbAttribs, &numConfigs);
    visualInfo = glXGetVisualFromFBConfig(_dpy, fbConfigs[0]);

    // Now create an X window
    winAttribs.event_mask = ExposureMask | VisibilityChangeMask | 
                            KeyPressMask | PointerMotionMask    |
                            StructureNotifyMask ;

    winAttribs.border_pixel = 0;
    winAttribs.bit_gravity = StaticGravity;
    winAttribs.colormap = XCreateColormap(_dpy, 
                                          RootWindow(_dpy, visualInfo->screen), 
                                          visualInfo->visual, AllocNone);
    winmask = CWBorderPixel | CWBitGravity | CWEventMask| CWColormap;

    _win = XCreateWindow(_dpy, DefaultRootWindow(_dpy), 20, 20,
                 32, 32, 0, 
                             visualInfo->depth, InputOutput,
                 visualInfo->visual, winmask, &winAttribs);

    //XMapWindow(_dpy, _win);//这一句注释掉就不会出现窗口的瞬间出现

    // Also create a new GL context for rendering
    GLint attribs[] = {
      GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
      GLX_CONTEXT_MINOR_VERSION_ARB, 5,
      0 };
    _ctx = glXCreateContextAttribsARB(_dpy, fbConfigs[0], 0, True, attribs);

    this->make_current();
    
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }

    //Testing 
    const GLubyte *s = glGetString(GL_VERSION);
    printf("GL Version = %s\n", s);    
}

void XGLContext::initialize()
{
    create_window();
}

void XGLContext::finalize()
{
    glXMakeCurrent(_dpy, None, NULL);
    glXDestroyContext(_dpy, _ctx);
    _ctx = NULL;

    XDestroyWindow(_dpy, _win);
    _win = (Window)NULL;

    XCloseDisplay(_dpy);
    _dpy = nullptr;
}

void XGLContext::make_current()
{
    glXMakeCurrent(_dpy, _win, _ctx);
}

void XGLContext::make_noncurrent()
{
    glXMakeCurrent(_dpy, None, NULL);
}

MED_IMG_END_NAMESPACE