#include "mi_model_interface.h"
#include "mi_observer_interface.h"

MED_IMG_BEGIN_NAMESPACE

IModel::IModel(): _is_changed(false) {

}

IModel::~IModel() {

}

void IModel::add_observer(ObserverPtr observer) {
    bool has_registered = false;
    auto it = _observers.begin();

    while (it != _observers.end()) {
        if (*it == observer) {
            has_registered = true;
            break;
        }

        ++it;
    }

    if (!has_registered) {
        _observers.push_back(observer);
    }
}

void IModel::delete_observer(ObserverPtr observer) {
    auto it = _observers.begin();

    while (it != _observers.end()) {
        if (*it == observer) {
            _observers.erase(it);
            break;
        }

        ++it;
    }
}

void IModel::notify(int code_id /*= 0*/) {
    if (_is_changed) {
        for (auto it = _observers.begin(); it != _observers.end(); ++it) {
            (*it)->update(code_id);
        }

        _is_changed = false;
    }
}

void IModel::set_changed() {
    _is_changed = true;
}

void IModel::reset_changed() {
    _is_changed = false;
}

bool IModel::has_changed() {
    return _is_changed;
}

void IModel::clear_observer() {
    _observers.clear();
}

MED_IMG_END_NAMESPACE