#ifndef MED_IMAGING_VOBSERVER_VOI_TABLE_H_
#define MED_IMAGING_VOBSERVER_VOI_TABLE_H_

#include "MedImgCommon/mi_observer_interface.h"
#include <QObject>

namespace medical_imaging
{
    class VOIModel;
}

class QNoduleObject : public QObject
{
    Q_OBJECT
public:
    QNoduleObject(QObject* parent =0);
    void add_nodule();
    void delete_nodule(int id);

signals:
    void nodule_added();
    void nodule_deleted(int id);
protected:
private:
};

class VOITableObserver : public medical_imaging::IObserver 
{
public:
    VOITableObserver();
    virtual ~VOITableObserver();
    void set_nodule_object(QNoduleObject* pNoduleObj);
    virtual void update();
protected:
private:
    QNoduleObject* m_pNoduleObject;
};

#endif