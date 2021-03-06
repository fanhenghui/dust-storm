

template<class T>
FIFOQueue<T>::~FIFOQueue() {
    std::deque<T>().swap(_container);
}

template<class T>
FIFOQueue<T>::FIFOQueue() {
}

template<class T>
void FIFOQueue<T>::clear() {
    _container.clear();
}

template<class T>
void FIFOQueue<T>::pop(T* elem) {
    *elem = std::move(_container.front());
    _container.pop_front();
}

template<class T>
void FIFOQueue<T>::push(const T& elem) {
    _container.push_back(elem);
}

template<class T>
bool FIFOQueue<T>::is_empty() const {
    return _container.empty();
}

template<class T>
size_t FIFOQueue<T>::size() const {
    return _container.size();
}
