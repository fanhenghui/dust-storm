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
    void add_observer(ObserverPtr pObserver);
    void delete_observer(ObserverPtr pObserver);
    void notify();
    void set_changed();
    void reset_changed();
    bool has_changed();

private:
    bool m_bIsChanged;
    std::vector<ObserverPtr> m_Observers;
};

MED_IMAGING_END_NAMESPACE

#endif