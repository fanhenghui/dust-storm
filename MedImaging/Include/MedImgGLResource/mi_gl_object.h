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

	void Reset()
	{
		boost::unique_lock<boost::mutex> locker(m_mutex);
		m_base = 0;
	}

	UIDType Tick()
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
	UIDType GetUID() const
	{
		return m_uid;
	}

    void SetType(const std::string& sType)
    {
        m_sType = sType;
    }

    std::string GetType() const
    {
        return m_sType;
    }

    std::string GetDescription() const
    {
        return m_sDescription;
    }

    void SetDescription(const std::string& sDes)
    {
        m_sDescription = sDes;
    }

	virtual void Initialize() = 0;
	virtual void Finalize() = 0;
	
private:
	UIDType m_uid;
    std::string m_sType;
    std::string m_sDescription;
};

MED_IMAGING_END_NAMESPACE

#endif
