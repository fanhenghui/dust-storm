
template<class ResourceType>
GLResourceManager<ResourceType>::GLResourceManager():m_pUIDGen(new GLUIDGenerator())
{
    
}

template<class ResourceType>
void GLResourceManager<ResourceType>::Update()
{
    boost::unique_lock<boost::mutex> locker(m_mutex);
    for (auto it = m_Discard.begin() ; it != m_Discard.end() ; )
    {
        if ((*it).use_count() > 1)
        {
            //std::cout << "Cant discard useless " << GetObjectDescription() << (*it)->GetUID() << std::endl;
            ++it;
        }
        else
        {
            (*it)->Finalize();
            it = m_Discard.erase(it);
        }
    }
}

template<class ResourceType>
void GLResourceManager<ResourceType>::RemoveAll()
{
    boost::unique_lock<boost::mutex> locker(m_mutex);
    for (auto it = m_Objects.begin() ; it != m_Objects.end() ; ++it)
    {
        m_Discard.push_back(it->second);
    }
    m_Objects.clear();
}

template<class ResourceType>
void GLResourceManager<ResourceType>::RemoveObject(UIDType uid)
{
    boost::unique_lock<boost::mutex> locker(m_mutex);
    auto it = m_Objects.find(uid);
    if (it != m_Objects.end())
    {
        m_Discard.push_back(it->second);
        m_Objects.erase(it);
    }
}

template<class ResourceType>
std::shared_ptr<ResourceType> GLResourceManager<ResourceType>::GetObject(UIDType uid)
{
    boost::unique_lock<boost::mutex> locker(m_mutex);
    auto it = m_Objects.find(uid);
    if (it == m_Objects.end())
    {
        return nullptr;
    }
    else
    {
        return it->second;
    }
}

template<class ResourceType>
std::shared_ptr<ResourceType> GLResourceManager<ResourceType>::CreateObject(UIDType &uid)
{
    boost::unique_lock<boost::mutex> locker(m_mutex);
    uid = m_pUIDGen->Tick();
    if (m_Objects.find(uid) != m_Objects.end())
    {
        GLRESOURCE_THROW_EXCEPTION("Generated UID invalid!");
    }

    std::shared_ptr<ResourceType> pResource(new ResourceType(uid));
    m_Objects[uid] = pResource;

    return pResource;
}

template<class ResourceType>
GLResourceManager<ResourceType>::~GLResourceManager()
{

}

template<class ResourceType>
std::string GLResourceManager<ResourceType>::GetObjectDescription()
{
    return typeid(ResourceType).name();
}
