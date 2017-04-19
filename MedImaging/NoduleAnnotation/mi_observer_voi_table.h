#ifndef MED_IMAGING_VOBSERVER_VOI_TABLE_H_
#define MED_IMAGING_VOBSERVER_VOI_TABLE_H_

#include "MedImgCommon/mi_observer_interface.h"
#include <QObject>

namespace MedImaging
{
    class VOIModel;
}

class QNoduleObject : public QObject
{
    Q_OBJECT
public:
    QNoduleObject(QObject* parent =0);
    void AddNodule();
    void DeleteNodule(int id);

signals:
    void addNodule();
    void deleteNodule(int id);
protected:
private:
};

class VOITableObserver : public MedImaging::IObserver 
{
public:
    VOITableObserver();
    virtual ~VOITableObserver();
    void SetNoduleObject(QNoduleObject* pNoduleObj);
    virtual void Update();
protected:
private:
    QNoduleObject* m_pNoduleObject;
};

#endif