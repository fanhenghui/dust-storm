#ifndef MED_IMG_GL_CONTEXT_H_
#define MED_IMG_GL_CONTEXT_H_

#include "MedImgGLResource/mi_gl_object.h"

MED_IMG_BEGIN_NAMESPACE

#ifdef WIN32

class MSGLContext : public GLObject
{
public:
    MSGLContext(UIDType uid);
    ~MSGLContext();

    virtual void initialize();
    virtual void finalize();
    
    void make_current();
    void make_noncurrent();
protected:
private: 
};

#else

class XGLContext : public GLObject
{
public:
    XGLContext(UIDType uid);
    ~XGLContext();

    virtual void initialize();
    virtual void finalize();
    
    void make_current();
    void make_noncurrent();
protected:
private: 
};

#endif

#ifdef WIN32
class GLResource_Export GLContext: public MSGLContext
#else
class GLResource_Export GLContext: public XGLContext
#endif
{
public:
#ifdef WIN32
    GLContext(UIDType uid):MSGLContext(uid)
    {

    };
#else
    GLContext(UIDType uid):XGLContext(uid)
    {

    };
#endif

    ~GLContext() {};
    
protected:
private:
};


MED_IMG_END_NAMESPACE

#endif

