#ifndef MEDIMGRESOURCE_GL_CONTEXT_H
#define MEDIMGRESOURCE_GL_CONTEXT_H

#include "GL/glew.h"
#include "glresource/mi_gl_object.h"
#include <map>

#ifdef WIN32

#else
#include "GL/glxew.h"
#endif

MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32

class GLResource_Export MSGLContext : public GLObject {
public:
    MSGLContext(UIDType uid);
    ~MSGLContext();

    virtual void initialize();
    virtual void finalize();

    void create_shared_context(int id);

    void make_current(int id = 0); // 0 is main context
    void make_noncurrent();

protected:
private:
};

#else

class GLResource_Export XGLContext : public GLObject {
public:
    XGLContext(UIDType uid);
    ~XGLContext();

    virtual void initialize();
    virtual void finalize();

    void create_shared_context(int id);

    void make_current(int id = 0); // 0 is main context
    void make_noncurrent();

private:
    void create_window();

private:
    GLXContext _ctx;
    std::map<int, GLXContext> _shared_ctx;
    Display* _dpy;
    XVisualInfo* _vis;
    Window _win;
};

#endif

#ifdef WIN32
class GLResource_Export GLContext : public MSGLContext
#else
class GLResource_Export GLContext : public XGLContext
#endif
{
public:
#ifdef WIN32
    GLContext(UIDType uid)
        : MSGLContext(uid) {

    };
#else
    GLContext(UIDType uid)
        : XGLContext(uid) {

    };
#endif

    ~GLContext() {};

protected:
private:
};

MED_IMG_END_NAMESPACE

#endif
