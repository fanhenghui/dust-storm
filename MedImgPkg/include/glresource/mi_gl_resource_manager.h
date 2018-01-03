#ifndef MEDIMGRESOURCE_GL_RESOURCE_MANAGER_H
#define MEDIMGRESOURCE_GL_RESOURCE_MANAGER_H

#include <sstream>
#include "boost/thread/mutex.hpp"
#include "glresource/mi_gl_object.h"
#include "log/mi_logger_util.h"

MED_IMG_BEGIN_NAMESPACE

template <class ResourceType> class GLResourceManager {
public:
    GLResourceManager();

    ~GLResourceManager();

    std::shared_ptr<ResourceType> create_object(UIDType& uid);

    std::shared_ptr<ResourceType> get_object(UIDType uid);

    void remove_object(UIDType uid);

    void remove_object(std::shared_ptr<ResourceType> obj);

    void remove_all();

    void update();

    std::string get_type() const;

private:
    std::map<UIDType, std::shared_ptr<ResourceType>> _objects;
    std::vector<std::shared_ptr<ResourceType>> _discard;
    std::unique_ptr<UIDGenerator> _uid_generator;
    boost::mutex _mutex;
};

#include "glresource/mi_gl_resource_manager.inl"

MED_IMG_END_NAMESPACE

#endif