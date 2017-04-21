
template<class ResourceType>
GLResourceManager<ResourceType>::GLResourceManager():m_pUIDGen(new GLUIDGenerator())
{
    
}

template<class ResourceType>
void GLResourceManager<ResourceType>::update()
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    for (auto it = m_Discard.begin() ; it != m_Discard.end() ; )
    {
        if ((*it).use_count() > 1)
        {
            //std::cout << "Cant discard useless " << GetObjectDescription() << (*it)->get_uid() << std::endl;
            ++it;
        }
        else
        {
            (*it)->finalize();
            it = m_Discard.erase(it);
        }
    }
}

template<class ResourceType>
void GLResourceManager<ResourceType>::remove_all()
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    for (auto it = m_Objects.begin() ; it != m_Objects.end() ; ++it)
    {
        m_Discard.push_back(it->second);
    }
    m_Objects.clear();
}

template<class ResourceType>
void GLResourceManager<ResourceType>::remove_object(UIDType uid)
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    auto it = m_Objects.find(uid);
    if (it != m_Objects.end())
    {
        m_Discard.push_back(it->second);
        m_Objects.erase(it);
    }
}

template<class ResourceType>
std::shared_ptr<ResourceType> GLResourceManager<ResourceType>::get_object(UIDType uid)
{
    boost::unique_lock<boost::mutex> locker(_mutex);
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
std::shared_ptr<ResourceType> GLResourceManager<ResourceType>::create_object(UIDType &uid)
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    uid = m_pUIDGen->tick();
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
std::string GLResourceManager<ResourceType>::get_type() const
{
    return typeid(ResourceType).name();
}
