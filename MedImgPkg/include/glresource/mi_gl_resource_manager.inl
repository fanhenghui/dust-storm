
template <class ResourceType>
GLResourceManager<ResourceType>::GLResourceManager()
    : _uid_generator(new UIDGenerator()) {}

template <class ResourceType> void GLResourceManager<ResourceType>::update() {
    boost::unique_lock<boost::mutex> locker(_mutex);

    for (auto it = _discard.begin(); it != _discard.end();) {
        if ((*it).use_count() > 1) {
            std::stringstream ss;
            ss << "Cant discard useless " << std::dynamic_pointer_cast<GLObject>(*it);
            LoggerUtil::log(MI_ERROR, "GLResource", ss.str());
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
        std::stringstream ss;
        ss << "remove useless " << std::dynamic_pointer_cast<GLObject>(it->second);
        LoggerUtil::log(MI_INFO, "GLResource", ss.str());
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
GLResourceManager<ResourceType>::create_object(const std::string& desc) {
    boost::unique_lock<boost::mutex> locker(_mutex);
    UIDType uid = _uid_generator->tick();

    if (_objects.find(uid) != _objects.end()) {
        GLRESOURCE_THROW_EXCEPTION("Generated UID invalid!");
    }

    std::shared_ptr<ResourceType> new_res(new ResourceType(uid));
    new_res->set_description(desc);
    _objects[uid] = new_res;

    std::stringstream ss;
    ss << "create " << std::dynamic_pointer_cast<GLObject>(new_res);
    LoggerUtil::log(MI_INFO, "GLResource", ss.str());
    return new_res;
}

template <class ResourceType>
GLResourceManager<ResourceType>::~GLResourceManager() {}

template <class ResourceType> 
std::string GLResourceManager<ResourceType>::get_specification(const std::string& split) {
    boost::mutex::scoped_lock locker(_mutex);
    std::stringstream ss;
    ss << get_type() << ": [" << split;
    float sum(0.0f);
    for (auto it = _objects.begin(); it != _objects.end(); ++it) {
        std::shared_ptr<ResourceType> ptr = it->second;
        if (ptr) {
            sum += ptr->memory_used();
            ss << "{" << ptr->get_description() << ", size: " << ptr->memory_used() << " KB }, " << split;
        }
    }

    ss << "{total size: " << sum << " KB }";
    ss << "]";
    return ss.str();
}