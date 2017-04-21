#ifndef MED_IMAGING_GL_RESOURCE_MANAGER_H_
#define MED_IMAGING_GL_RESOURCE_MANAGER_H_

#include "boost/thread/mutex.hpp"
#include "MedImgGLResource/mi_gl_object.h"

MED_IMAGING_BEGIN_NAMESPACE

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
    std::map<UIDType, std::shared_ptr<ResourceType>> m_Objects;
    std::vector<std::shared_ptr<ResourceType>> m_Discard;
    std::unique_ptr<GLUIDGenerator> m_pUIDGen;
    boost::mutex _mutex;
};

#include "MedImgGLResource/mi_gl_resource_manager.inl"

MED_IMAGING_END_NAMESPACE

#endif