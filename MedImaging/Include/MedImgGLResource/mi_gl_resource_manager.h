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

    std::string GetObjectDescription();

    std::shared_ptr<ResourceType> CreateObject(UIDType &uid);

    std::shared_ptr<ResourceType> GetObject(UIDType uid);

    void RemoveObject(UIDType uid);

    void RemoveAll();

    void Update();

private:
    std::map<UIDType, std::shared_ptr<ResourceType>> m_Objects;
    std::vector<std::shared_ptr<ResourceType>> m_Discard;
    std::unique_ptr<GLUIDGenerator> m_pUIDGen;
    boost::mutex m_mutex;
};

#include "MedImgGLResource/mi_gl_resource_manager.inl"

MED_IMAGING_END_NAMESPACE

#endif