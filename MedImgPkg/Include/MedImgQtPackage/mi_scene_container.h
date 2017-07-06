#ifndef MED_IMG_SCENE_CONTAINER_H_
#define MED_IMG_SCENE_CONTAINER_H_

#include "MedImgQtPackage/mi_qt_package_export.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>

#include "boost/thread/mutex.hpp"

namespace medical_imaging
{
    class SceneBase;
    class IMouseOp;
    class GraphicItemBase;
}

typedef std::shared_ptr<medical_imaging::IMouseOp> IMouseOpPtr;
typedef std::vector<IMouseOpPtr> IMouseOpPtrCollection;
typedef std::shared_ptr<medical_imaging::GraphicItemBase> GraphicItemPtr;

class Graphics2DScene : public QGraphicsScene
{
    Q_OBJECT
public:
    Graphics2DScene(QObject* parent = 0);
    virtual ~Graphics2DScene();
};

//这里我采用了Qt的QGraphicItem + QGraphicScene + QGraphicView 框架来做，原因是Qt自带了一套图元和鼠标交互的逻辑（如鼠标选择拖动圆 长方形 文字等）
//所以二位图元交互的逻辑为：Qt改变该图元 -> 触发图元修改的逻辑（业务逻辑）
//注意的是：Qt只有碰撞和拖动逻辑，因此类似改变图元形状这样的逻辑就需要自己扩展
//目前Qt的碰撞热区是包含轮廓内的区域，如圆，则是圆内而非仅仅是圆圈
class QtPackage_Export SceneContainer : public QGraphicsView
{
    Q_OBJECT
public:
    SceneContainer(QWidget* parent = 0);
    virtual ~SceneContainer();

    void set_name(const std::string& des);
    std::string get_name() const;

    void set_scene(std::shared_ptr<medical_imaging::SceneBase> scene);
    std::shared_ptr<medical_imaging::SceneBase> get_scene();
    void update_scene();

    void add_item(GraphicItemPtr item);
    void clear();

    void register_mouse_operation(IMouseOpPtr mouse_op , Qt::MouseButtons buttons , Qt::KeyboardModifier keyboard_modifier);
    void register_mouse_operation(IMouseOpPtrCollection mouse_ops , Qt::MouseButtons buttons , Qt::KeyboardModifier keyboard_modifier);
    void register_mouse_wheel_operation(IMouseOpPtr mouse_op);
    void register_mouse_wheel_operation(IMouseOpPtrCollection mouse_ops);
    IMouseOpPtrCollection get_mouse_operation(Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier);

signals:
    void focus_in_scene();
    void focus_out_scene();

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

    virtual void drawBackground(QPainter *painter, const QRectF &rect);
    virtual void drawForeground(QPainter *painter, const QRectF &rect);

    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent( QMouseEvent *event );
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseDoubleClickEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent *);
    virtual void keyPressEvent(QKeyEvent *key);
    virtual void keyReleaseEvent(QKeyEvent *key);

    virtual void focusInEvent(QFocusEvent *event);
    virtual void focusOutEvent(QFocusEvent *event);

private slots:
    void slot_mouse_click();

private:
    bool get_mouse_operation_i(IMouseOpPtrCollection& op);
    bool no_graphics_item_grab_i();

private:
    QGLWidget* _inner_gl_widget;
    Graphics2DScene* _inner_graphic_scene;

    std::shared_ptr<medical_imaging::SceneBase> _scene;

    boost::mutex _mutex;

    //Mouse interaction
    std::map<int , IMouseOpPtrCollection> _mouse_ops;
    IMouseOpPtrCollection _mouse_wheel_ops;
    Qt::MouseButtons _buttons;
    Qt::KeyboardModifiers _modifiers;
    QPointF _pre_point;

    int _mouse_press_time;
    int _mouse_release_time;
    Qt::MouseButtons _buttons_pre_press;

    std::vector<GraphicItemPtr> _graphic_items;
};

#endif