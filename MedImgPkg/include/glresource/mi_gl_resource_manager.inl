
template <class ResourceType>
GLResourceManager<ResourceType>::GLResourceManager()
    : _uid_generator(new GLUIDGenerator()) {}

template <class ResourceType> void GLResourceManager<ResourceType>::update() {
    boost::unique_lock<boost::mutex> locker(_mutex);

    for (auto it = _discard.begin(); it != _discard.end();) {
        if ((*it).use_count() > 1) {
            // std::cout << "Cant discard useless " << GetObjectDescription() <<
            // (*it)->get_uid() << std::endl;
            ++it;
        } else {
            (*it)->finalize();
            it = _discard.erase(it);
        }
    }
}

template <class ResourceType>
void GLResourceManager<ResourceType>::remove_all() {
    boost::unique_lock<boost::mutex> locker(_mutex);

    for (auto it = _objects.begin(); it != _objects.end(); ++it) {
        _discard.push_back(it->second);
    }

    _objects.clear();
}

template <class ResourceType>
void GLResourceManager<ResourceType>::remove_object(UIDType uid) {
    boost::unique_lock<boost::mutex> locker(_mutex);
    auto it = _objects.find(uid);

    if (it != _objects.end()) {
        std::cout << "remove useless " << it->second->get_description() << uid
                  << std::endl;
        _discard.push_back(it->second);
        _objects.erase(it);
    }
}

template <class ResourceType>
void GLResourceManager<ResourceType>::remove_object(
    std::shared_ptr<ResourceType> obj) {
    this->remove_object(obj->get_uid());
}

template <class ResourceType>
std::shared_ptr<ResourceType>
GLResourceManager<ResourceType>::get_object(UIDType uid) {
    boost::unique_lock<boost::mutex> locker(_mutex);
    auto it = _objects.find(uid);

    if (it == _objects.end()) {
        return nullptr;
    } else {
        return it->second;
    }
}

template <class ResourceType>
std::shared_ptr<ResourceType>
GLResourceManager<ResourceType>::create_object(UIDType& uid) {
    boost::unique_lock<boost::mutex> locker(_mutex);
    uid = _uid_generator->tick();

    if (_objects.find(uid) != _objects.end()) {
        GLRESOURCE_THROW_EXCEPTION("Generated UID invalid!");
    }

    std::shared_ptr<ResourceType> pResource(new ResourceType(uid));
    _objects[uid] = pResource;

    return pResource;
}

template <class ResourceType>
GLResourceManager<ResourceType>::~GLResourceManager() {}

template <class ResourceType>
std::string GLResourceManager<ResourceType>::get_type() const {
    return typeid(ResourceType).name();
}