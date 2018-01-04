#ifndef MEDIMGRESOURCE_GL_OBJECT_H_
#define MEDIMGRESOURCE_GL_OBJECT_H_

#include "GL/glew.h"
#include "boost/thread/mutex.hpp"
#include "glresource/mi_gl_resource_export.h"
#include "util/mi_uid.h"

MED_IMG_BEGIN_NAMESPACE

class GLObject {
public:
    GLObject(UIDType uid, const std::string& type) : _uid(uid), _type(type) {}
    virtual  ~GLObject() {}

    UIDType get_uid() const {
        return _uid;
    }

    void set_type(const std::string& type) {
        _type = type;
    }

    std::string get_type() const {
        return _type;
    }

    std::string get_description() const {
        return _description;
    }

    void set_description(const std::string& des) {
        _description = des;
    }

    friend std::ostream& operator << (std::ostream& strm ,const GLObject& obj) {
        strm << "GLOBJ type: " << obj.get_type() << ", uid: " << obj.get_uid() << ", des: " << obj.get_description();
        return strm;
    }

    friend std::ostream& operator << (std::ostream& strm ,const std::shared_ptr<GLObject>& obj) {
        strm << "GLOBJ type: " << obj->get_type() << ", uid: " << obj->get_uid() << ", des: " << obj->get_description();
        return strm;
    }

    virtual void initialize() = 0;
    virtual void finalize() = 0;

private:
    UIDType _uid;
    std::string _type;
    std::string _description;
};

MED_IMG_END_NAMESPACE

#endif
