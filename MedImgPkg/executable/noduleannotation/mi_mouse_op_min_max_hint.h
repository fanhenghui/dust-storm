#ifndef MED_IMG_MOUSE_OP_MIN_MAX_HINT_H_
#define MED_IMG_MOUSE_OP_MIN_MAX_HINT_H_

#include "qtpackage/mi_mouse_op_interface.h"
#include "arithmetic/mi_point3.h"
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

MED_IMG_BEGIN_NAMESPACE


class MouseOpMinMaxHint : public IMouseOp
{
public:
    MouseOpMinMaxHint();
    virtual ~MouseOpMinMaxHint();

    virtual void press(const QPointF& pt);
    virtual void move(const QPointF& pt);
    virtual void release(const QPointF& pt);
    virtual void double_click(const QPointF& pt);
    virtual void wheel_slide(int);

    void set_min_max_hint_object(QMinMaxHintObject* obj);
protected:
private:
    QMinMaxHintObject* _min_max_hint_object;
};

MED_IMG_END_NAMESPACE

#endif