#ifndef MEDIMGUTIL_MI_MESSAGE_QUEUE_H
#define MEDIMGUTIL_MI_MESSAGE_QUEUE_H

#include "util/mi_util_export.h"
#include "util/mi_exception.h"
#include "util/mi_util_logger.h"

#include <vector>
#include <deque>
#include <limits>

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/condition.hpp"
#include "boost/noncopyable.hpp"


MED_IMG_BEGIN_NAMESPACE

template<class T>
class Queue {
public:
    virtual ~Queue() {};
    virtual size_t size() const = 0;
    virtual bool is_empty() const = 0;
    virtual void push(const T&) = 0;
    virtual void pop(T*) = 0;
    virtual void clear() = 0;
};

template<class T>
class FIFOQueue : public Queue<T> {
public:
    FIFOQueue();
    virtual ~FIFOQueue();

    virtual size_t size() const;
    virtual bool is_empty() const;
    virtual void push(const T&);
    virtual void pop(T*);
    virtual void clear();
private:
    std::deque<T> _container;
};

template<class T>
class LIFOQueue : public Queue<T> {
public:
    LIFOQueue();
    virtual ~LIFOQueue();

    virtual size_t size() const;
    virtual bool is_empty() const;
    virtual void push(const T&);
    virtual void pop(T*);
    virtual void clear();
private:
    std::vector<T> _container;
};

template<class T , class TQueue = FIFOQueue<T>>
class MessageQueue : public boost::noncopyable {
public:
    MessageQueue(): _is_activated(false) {

    }

    ~MessageQueue() {
    }

    void wait_to_push(boost::mutex::scoped_lock& locker , int time_wait_limit) {
        auto time = boost::get_system_time() +
                    boost::posix_time::milliseconds(time_wait_limit);

        while (is_full()) {
            if (!_condition_write.timed_wait(locker , time)) {
                MI_UTIL_LOG(MI_FATAL) << "message queue time out to push.";
                UTIL_THROW_EXCEPTION("message queue time out to push.");
            }
        }
    }

    void wait_to_pop(boost::mutex::scoped_lock& locker , int time_wait_limit) {
        auto time = boost::get_system_time() +
                    boost::posix_time::milliseconds(time_wait_limit);

        while (is_empty()) {
            if (!_condition_read.timed_wait(locker , time)) {
                MI_UTIL_LOG(MI_FATAL) << "message queue time out to pop.";
                UTIL_THROW_EXCEPTION("message time out to pop.");
            }
        }

    }

    size_t capacity() const {
        return _DEFAULT_CAPACITY;
    }

    size_t size() const {
        return _container.size();
    }

    bool is_full() const {
        return _DEFAULT_CAPACITY <= _container.size();
    }

    bool is_empty() const {
        return _container.is_empty();
    }

    void activate() {
        _is_activated = true;
    }

    void deactivate() {
        if (_is_activated) {
            boost::mutex::scoped_lock locker(_mutex);

            _is_activated = false;
            _condition_read.notify_all();
            _condition_write.notify_all();
        }
    }

    bool is_activated() const {
        return _is_activated;
    }

    void push(const T& msg) {
        boost::mutex::scoped_lock locker(_mutex);

        wait_to_push(locker , _DEFAULT_TIME_WAIT_LIMIT);

        if (!is_activated()) {
            MI_UTIL_LOG(MI_FATAL) << "message queue is not activated.";
            UTIL_THROW_EXCEPTION("message queue is not activated.");
        }

        _container.push(msg);

        _condition_read.notify_one();
    }

    void pop(T* msg) {
        boost::mutex::scoped_lock locker(_mutex);

        wait_to_pop(locker , _DEFAULT_TIME_WAIT_LIMIT);

        if (!is_activated()) {
            MI_UTIL_LOG(MI_FATAL) << "message queue is not activated.";
            UTIL_THROW_EXCEPTION("message queue is not activated.");
        }

        _container.pop(msg);

        _condition_write.notify_one();
    }

protected:
private:
    const static size_t _DEFAULT_CAPACITY = 2000;
    const static int _DEFAULT_TIME_WAIT_LIMIT = 0X7FFFFFFF;

    TQueue _container;
    bool _is_activated;

    mutable boost::mutex _mutex;
    boost::condition _condition_read;
    boost::condition _condition_write;
};

#include "mi_message_queue.inl"

MED_IMG_END_NAMESPACE


#endif