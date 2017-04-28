#ifndef MED_IMAGING_COMMON_MODEL_H_
#define MED_IMAGING_COMMON_MODEL_H_


#include "MedImgCommon/mi_common_stdafx.h"
#include <vector>

MED_IMAGING_BEGIN_NAMESPACE

class IObserver;
typedef std::shared_ptr<IObserver> ObserverPtr;

class Common_Export IModel
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

MED_IMAGING_END_NAMESPACE

#endif