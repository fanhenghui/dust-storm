#ifndef MED_IMG_UTIL_MODEL_H_
#define MED_IMG_UTIL_MODEL_H_

#include "MedImgUtil/mi_util_export.h"
#include <vector>

MED_IMG_BEGIN_NAMESPACE

class IObserver;
typedef std::shared_ptr<IObserver> ObserverPtr;

class Util_Export IModel
{
public:
    IModel();
    virtual ~IModel();

    void add_observer(ObserverPtr observer);
    void delete_observer(ObserverPtr observer);
    void clear_observer();

    void notify(int code_id = 0);

    void set_changed();
    void reset_changed();
    bool has_changed();

private:
    bool _is_changed;
    std::vector<ObserverPtr> _observers;
};

MED_IMG_END_NAMESPACE

#endif