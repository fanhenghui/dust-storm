#ifndef MED_IMAGING_GL_OBJECT_H_
#define MED_IMAGING_GL_OBJECT_H_

#include "gl/glew.h"
#include "boost/thread/mutex.hpp"
#include "MedImgGLResource/mi_gl_resource_stdafx.h"

MED_IMAGING_BEGIN_NAMESPACE

    typedef unsigned long UIDType;

class GLUIDGenerator
{
public:
    GLUIDGenerator():m_base(0)
    {
    }

    ~GLUIDGenerator()
    {
        m_base = 0;
    }

    void reset()
    {
        boost::unique_lock<boost::mutex> locker(m_mutex);
        m_base = 0;
    }

    UIDType tick()
    {
        boost::unique_lock<boost::mutex> locker(m_mutex);
        return m_base++;
    }

private:
    UIDType m_base;
    boost::mutex m_mutex;
};

class GLResource_Export GLObject
{
public:
    GLObject(UIDType uid):m_uid(uid)
    {

    }
    ~GLObject()
    {

    }
    UIDType get_uid() const
    {
        return m_uid;
    }

    void set_type(const std::string& sType)
    {
        m_sType = sType;
    }

    std::string get_type() const
    {
        return m_sType;
    }

    std::string get_description() const
    {
        return m_sDescription;
    }

    void set_description(const std::string& sDes)
    {
        m_sDescription = sDes;
    }

    virtual void initialize() = 0;
    virtual void finalize() = 0;

private:
    UIDType m_uid;
    std::string m_sType;
    std::string m_sDescription;
};

MED_IMAGING_END_NAMESPACE

#endif
