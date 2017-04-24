#ifndef MED_IMAGING_SCENE_CONTAINER_H_
#define MED_IMAGING_SCENE_CONTAINER_H_

#include "MedImgQtWidgets/mi_qt_widgets_stdafx.h"
#include <QtOpenGL/qgl.h>
#include "qt/qpoint.h"
#include "boost/thread/mutex.hpp"

namespace medical_imaging
{
    class SceneBase;
    class PainterBase;
    class IMouseOp;
}

class QPaintEvent;
class QMouseEvent;
class QKeyEvent;

class SharedWidget;

typedef std::shared_ptr<medical_imaging::IMouseOp> IMouseOpPtr;
typedef std::vector<IMouseOpPtr> IMouseOpPtrCollection;
typedef std::shared_ptr<medical_imaging::PainterBase> PainterPtr;

class QtWidgets_Export SceneContainer : public QGLWidget
{
    Q_OBJECT
public:
    SceneContainer(SharedWidget* shared , QWidget* parent = 0);
    virtual ~SceneContainer();

    void set_name(const std::string& des);
    std::string get_name() const;

    void set_scene(std::shared_ptr<medical_imaging::SceneBase> scene);
    std::shared_ptr<medical_imaging::SceneBase> get_scene();

    void register_mouse_operation(IMouseOpPtr mouse_op , Qt::MouseButtons buttons , Qt::KeyboardModifier keyboard_modifier);
    void register_mouse_operation(IMouseOpPtrCollection mouse_ops , Qt::MouseButtons buttons , Qt::KeyboardModifier keyboard_modifier);
    void register_mouse_wheel_operation(IMouseOpPtr mouse_op);
    void register_mouse_wheel_operation(IMouseOpPtrCollection mouse_ops);
    IMouseOpPtrCollection get_mouse_operation(Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier);

    void register_key_operation();

    void add_painter_list(std::vector<PainterPtr> painters);

signals:
    void focus_in_scene();
    void focus_out_scene();

protected:
    //////////////////////////////////////////////////////////////////////////
    //Qt virtual function for interaction 
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintEvent(QPaintEvent* painter);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent( QMouseEvent *event );
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseDoubleClickEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent *);
    virtual void keyPressEvent(QKeyEvent *key);
    virtual void keyReleaseEvent(QKeyEvent *key);
    virtual void focusInEvent(QFocusEvent *event);
    virtual void focusOutEvent(QFocusEvent *event);
    ////////////////////////////////////////////////////////////////////////// 

    void render_to_back_i();

private slots:
    void slot_mouse_click();

private:
    bool get_mouse_operation_i(IMouseOpPtrCollection& op);

protected:
    std::shared_ptr<medical_imaging::SceneBase> _scene;
    std::vector<PainterPtr> _painters;
    std::map<int , IMouseOpPtrCollection> _mouse_ops;
    IMouseOpPtrCollection _mouse_wheel_ops;
    Qt::MouseButtons _buttons;
    Qt::KeyboardModifiers _modifiers;
    QPoint _pre_point;

    std::unique_ptr<QPixmap> _pixel_map;
    boost::mutex _mutex;

    int _mouse_press_time;
    int _mouse_release_time;
    Qt::MouseButtons _buttons_pre_press;
};

#endif