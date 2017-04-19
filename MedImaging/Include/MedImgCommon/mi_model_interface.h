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
    void AddObserver(ObserverPtr pObserver);
    void DeleteObserver(ObserverPtr pObserver);
    void NotifyAllObserver();
    void SetChanged();
    void CleanChanged();
    bool HasChanged();

private:
    bool m_bIsChanged;
    std::vector<ObserverPtr> m_Observers;
};

MED_IMAGING_END_NAMESPACE

#endif