
template<class ResourceType>
GLResourceManager<ResourceType>::GLResourceManager():_uid_generator(new GLUIDGenerator())
{
    
}

template<class ResourceType>
void GLResourceManager<ResourceType>::update()
{
    std::unique_lock<std::mutex> locker(_mutex);
    for (auto it = _discard.begin() ; it != _discard.end() ; )
    {
        if ((*it).use_count() > 1)
        {
            //std::cout << "Cant discard useless " << GetObjectDescription() << (*it)->get_uid() << std::endl;
            ++it;
        }
        else
        {
            (*it)->finalize();
            it = _discard.erase(it);
        }
    }
}

template<class ResourceType>
void GLResourceManager<ResourceType>::remove_all()
{
    std::unique_lock<std::mutex> locker(_mutex);
    for (auto it = _objects.begin() ; it != _objects.end() ; ++it)
    {
        _discard.push_back(it->second);
    }
    _objects.clear();
}

template<class ResourceType>
void GLResourceManager<ResourceType>::remove_object(UIDType uid)
{
    std::unique_lock<std::mutex> locker(_mutex);
    auto it = _objects.find(uid);
    if (it != _objects.end())
    {
        _discard.push_back(it->second);
        _objects.erase(it);
    }
}

template<class ResourceType>
std::shared_ptr<ResourceType> GLResourceManager<ResourceType>::get_object(UIDType uid)
{
    std::unique_lock<std::mutex> locker(_mutex);
    auto it = _objects.find(uid);
    if (it == _objects.end())
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
    std::unique_lock<std::mutex> locker(_mutex);
    uid = _uid_generator->tick();
    if (_objects.find(uid) != _objects.end())
    {
        return nullptr;
        //GLRESOURCE_THROW_EXCEPTION("Generated UID invalid!");
    }

    std::shared_ptr<ResourceType> pResource(new ResourceType(uid));
    _objects[uid] = pResource;

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
