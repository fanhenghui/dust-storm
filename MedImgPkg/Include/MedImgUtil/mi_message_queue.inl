

template<class T>
FIFOQueue<T>::~FIFOQueue()
{
    std::deque<T>().swap(_container);
}

template<class T>
FIFOQueue<T>::FIFOQueue()
{
}

template<class T>
void FIFOQueue<T>::clear()
{
    _container.clear();
}

template<class T>
void FIFOQueue<T>::pop(T* elem)
{
    *elem = std::move(_container.front());
    _container.pop_front();
}

template<class T>
void FIFOQueue<T>::push(const T& elem)
{
    _container.push_back(elem);
}

template<class T>
bool FIFOQueue<T>::is_empty() const
{
    return _container.empty();
}

template<class T>
size_t FIFOQueue<T>::size() const
{
    return _container.size();
}


template<class T>
LIFOQueue<T>::~LIFOQueue()
{
    std::vector<T>().swap(_container);
}

template<class T>
LIFOQueue<T>::LIFOQueue()
{
}

template<class T>
void LIFOQueue<T>::clear()
{
    _container.clear();
}

template<class T>
void LIFOQueue<T>::pop(T* elem)
{
    *elem = std::move(_container.front());
    _container.pop_back(elem);
}

template<class T>
void LIFOQueue<T>::push(const T& elem)
{
    _container.push_back(elem);
}

template<class T>
bool LIFOQueue<T>::is_empty() const
{
    return _container.empty();
}

template<class T>
size_t LIFOQueue<T>::size() const
{
    return _container.size();
}

