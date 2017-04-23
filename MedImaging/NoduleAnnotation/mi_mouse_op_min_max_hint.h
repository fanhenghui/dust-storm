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
    void trigger(const std::string& s);

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
    virtual void press(const QPoint& pt);
    virtual void move(const QPoint& pt);
    virtual void release(const QPoint& pt);
    virtual void double_click(const QPoint& pt);
    void set_min_max_hint_object(QMinMaxHintObject* obj);
protected:
private:
    QMinMaxHintObject* _min_max_hint_object;
};

MED_IMAGING_END_NAMESPACE

#endif