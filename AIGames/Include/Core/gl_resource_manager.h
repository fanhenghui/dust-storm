#ifndef MED_IMAGING_GL_RESOURCE_MANAGER_H_
#define MED_IMAGING_GL_RESOURCE_MANAGER_H_

#include <mutex>
#include <vector>
#include <map>
#include "Core/gl_object.h"



template<class ResourceType>
class GLResourceManager
{
public:
    GLResourceManager();

    ~GLResourceManager();

    std::shared_ptr<ResourceType> create_object(UIDType &uid);

    std::shared_ptr<ResourceType> get_object(UIDType uid);

    void remove_object(UIDType uid);

    void remove_all();

    void update();

    std::string get_type() const;

private:
    std::map<UIDType, std::shared_ptr<ResourceType>> _objects;
    std::vector<std::shared_ptr<ResourceType>> _discard;
    std::unique_ptr<GLUIDGenerator> _uid_generator;
    std::mutex _mutex;
};

#include "Core/gl_resource_manager.inl"



#endif