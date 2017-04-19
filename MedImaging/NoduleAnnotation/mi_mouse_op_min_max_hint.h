#ifndef MED_IMAGING_MOUSE_OP_MIN_MAX_HINT_H_
#define MED_IMAGING_MOUSE_OP_MIN_MAX_HINT_H_

#include "MedImgQtWidgets/mi_mouse_op_interface.h"
#include "MedImgArithmetic/mi_point3.h"
#include <QObject>

class QMinMaxHintObject : public QObject
{
    Q_OBJECT
public:
    QMinMaxHintObject(QObject* parent = 0);
    ~QMinMaxHintObject();
    void Triggered(const std::string& s);

signals:
    void triggered(const std::string& s);

protected:
private:
};

MED_IMAGING_BEGIN_NAMESPACE


class MouseOpMinMaxHint : public IMouseOp
{
public:
    MouseOpMinMaxHint();
    virtual ~MouseOpMinMaxHint();
    virtual void Press(const QPoint& pt);
    virtual void Move(const QPoint& pt);
    virtual void Release(const QPoint& pt);
    virtual void DoubleClick(const QPoint& pt);
    void SetMinMaxHintObject(QMinMaxHintObject* pObj);
protected:
private:
    QMinMaxHintObject* m_pMinMaxHintObject;
};

MED_IMAGING_END_NAMESPACE

#endif