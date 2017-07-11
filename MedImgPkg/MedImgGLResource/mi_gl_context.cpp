#include "mi_gl_context.h"

#ifdef WIN32

#else

#include <stdio.h>
#include <stdlib.h>

#endif

MED_IMG_BEGIN_NAMESPACE

XGLContext::XGLContext(UIDType uid):GLObject(uid),_dpy(nullptr),_vis(nullptr),_ctx(NULL),_win((Window)NULL)
{

}

XGLContext::~XGLContext()
{

}

void XGLContext::create_window()
{
    /// \Open display
    _dpy = XOpenDisplay(NULL);
    if (nullptr == _dpy) {
        GLRESOURCE_THROW_EXCEPTION("Failed to open X display!\n");
    }

    /// \Choose visual
    //get visual info RGB
    int attributes_list[32];
    int n = 0;
    attributes_list[n++] = GLX_RGBA;
    attributes_list[n++] = GLX_RED_SIZE;
    attributes_list[n++] = 1;
    attributes_list[n++] = GLX_GREEN_SIZE;
    attributes_list[n++] = 1;
    attributes_list[n++] = GLX_BLUE_SIZE;
    attributes_list[n++] = 1;
    //if (GLUT_WIND_HAS_ALPHA(mode)) {
          attributes_list[n++] = GLX_ALPHA_SIZE;
          attributes_list[n++] = 1;
    //}
    //if (GLUT_WIND_IS_DOUBLE(mode)) {
      attributes_list[n++] = GLX_DOUBLEBUFFER;
    //}
    //if (GLUT_WIND_IS_STEREO(mode)) {//立体显示的时候要设置
    //  attributes_list[n++] = GLX_STEREO;
    //}
    //if (GLUT_WIND_HAS_DEPTH(mode)) {
      attributes_list[n++] = GLX_DEPTH_SIZE;
      attributes_list[n++] = 1;
    //}
    //if (GLUT_WIND_HAS_STENCIL(mode)) {
      attributes_list[n++] = GLX_STENCIL_SIZE;
      attributes_list[n++] = 1;
    //}
    //if (GLUT_WIND_HAS_ACCUM(mode)) {
    //  attributes_list[n++] = GLX_ACCUM_RED_SIZE;
    //  attributes_list[n++] = 1;
    //  attributes_list[n++] = GLX_ACCUM_GREEN_SIZE;
    //  attributes_list[n++] = 1;
    //  attributes_list[n++] = GLX_ACCUM_BLUE_SIZE;
    //  attributes_list[n++] = 1;
    //  if (GLUT_WIND_HAS_ALPHA(mode)) {
    //    attributes_list[n++] = GLX_ACCUM_ALPHA_SIZE;
    //    attributes_list[n++] = 1;
    //  }
    //}
    attributes_list[n] = (int) None;//end tag
    
    _vis = glXChooseVisual(_dpy, DefaultScreen(_dpy), attributes_list);
    if (nullptr == _vis) {
        GLRESOURCE_THROW_EXCEPTION("Failed to choose visual!\n");
    }

    /// \Create OpenGL context
    _ctx = glXCreateContext(_dpy, _vis, 0, GL_TRUE);
    if ( nullptr == _ctx) {
        GLRESOURCE_THROW_EXCEPTION("Failed to create OpenGL context!\n");
    }

    /// \Create window
    Colormap colormap = XCreateColormap(_dpy, RootWindow(_dpy, _vis->screen), _vis->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.background_pixmap = None;
    swa.border_pixel = 0;
    swa.colormap = colormap;
    unsigned long attribMask =  CWBackPixmap | CWBorderPixel | CWColormap | CWEventMask;
    _win = XCreateWindow(
        _dpy,
        RootWindow(_dpy, _vis->screen),
        0, 0, 64, 64,
        0, _vis->depth, InputOutput, _vis->visual,
        attribMask,
        &swa
    );

    // Get Version info
    int major = 0;
    int minor = 0;
    glXQueryVersion(_dpy, &major, &minor);
    printf("Supported GLX version - %d.%d\n", major, minor);   

    //if(major == 1 && minor < 2)
    //{
    //    printf("ERROR: GLX 1.2 or greater is necessary\n");
    //    XCloseDisplay(_dpy);
    //    exit(0);
    //}

    //XMapWindow(_dpy, _win);//这一句注释掉就不会出现窗口的瞬间出现,而始终保持隐藏

    this->make_current();
    
    glewExperimental = GL_TRUE;

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


void XGLContext::create_shared_context(int id)
{
    if(_ctx == NULL){
        GLRESOURCE_THROW_EXCEPTION("main context is NULL!");
    }

    if(_shared_ctx.find(id) != _shared_ctx.end()){
        GLRESOURCE_THROW_EXCEPTION("Create the same shared context id");
    }

    GLXContext ctx = glXCreateContext(_dpy, _vis, _ctx, GL_TRUE);
    if ( nullptr == _ctx) {
        GLRESOURCE_THROW_EXCEPTION("Failed to create OpenGL context!\n");
    }

    _shared_ctx[id] = ctx;
    
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
    for(auto it = _shared_ctx.begin() ; it != _shared_ctx.end() ; ++it){
        glXDestroyContext(_dpy, it->second);
    }
    _shared_ctx.clear();

    XDestroyWindow(_dpy, _win);
    _win = (Window)NULL;

    XCloseDisplay(_dpy);
    _dpy = nullptr;
}

void XGLContext::make_current(int id )
{
    if(0 == id){
        glXMakeCurrent(_dpy, _win, _ctx);
    }
    else{
        auto it = _shared_ctx.find(id);
        if(it != _shared_ctx.end()){
            glXMakeCurrent(_dpy, _win, it->second);
        }
        else{
            GLRESOURCE_THROW_EXCEPTION("cant find such shared context");
        }
    }
    
}

void XGLContext::make_noncurrent()
{
    glXMakeCurrent(_dpy, None, NULL);
}

MED_IMG_END_NAMESPACE