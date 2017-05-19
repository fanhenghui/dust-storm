#ifndef MED_IMAGING_GL_OBJECT_H_
#define MED_IMAGING_GL_OBJECT_H_

#include "Ext/gl/glew.h"
#include <mutex>


typedef unsigned long UIDType;

class GLUIDGenerator
{
public:
    GLUIDGenerator():_base(0)
    {
    }

    ~GLUIDGenerator()
    {
        _base = 0;
    }

    void reset()
    {
        std::unique_lock<std::mutex> locker(_mutex);
        _base = 0;
    }

    UIDType tick()
    {
        std::unique_lock<std::mutex> locker(_mutex);
        
        return _base++;
    }

private:
    UIDType _base;
    std::mutex _mutex;
};

class  GLObject
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

    void set_type(const std::string& type)
    {
        m_type = type;
    }

    std::string get_type() const
    {
        return m_type;
    }

    std::string get_description() const
    {
        return _description;
    }

    void set_description(const std::string& des)
    {
        _description = des;
    }

    virtual void initialize() = 0;
    virtual void finalize() = 0;

private:
    UIDType m_uid;
    std::string m_type;
    std::string _description;
};



#endif
