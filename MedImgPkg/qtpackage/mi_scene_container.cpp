#include "mi_scene_container.h"

#include <QMouseEvent>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QTimer>
#include <QApplication>
//#include <QDebug>
#include <QMouseEvent>

#include "renderalgo/mi_scene_base.h"

#include "mi_shared_widget.h"
#include "mi_mouse_op_interface.h"
#include "mi_graphic_item_base.h"

using namespace medical_imaging;


Graphics2DScene::Graphics2DScene(QObject* parent /*= 0*/):
QGraphicsScene(parent)
{

}

Graphics2DScene::~Graphics2DScene()
{

}

//////////////////////////////////////////////////////////////////////////

SceneContainer::SceneContainer(QWidget* parent /*= 0*/):
        QGraphicsView(parent),
        _buttons(Qt::NoButton),
        _modifiers(Qt::NoModifier),
        _mouse_press_time(0),
        _mouse_release_time(0),
        _pre_point(0,0),
        _buttons_pre_press(Qt::NoButton),
        _double_click_interval(QApplication::doubleClickInterval())
{
    _inner_graphic_scene = new Graphics2DScene();
    this->setScene(_inner_graphic_scene);

    _inner_gl_widget = new QGLWidget(0 , SharedWidget::instance() );
    this->setViewport(_inner_gl_widget );

    //this->setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);

    this->setMouseTracking(false);
}

SceneContainer::~SceneContainer()
{

}

void SceneContainer::drawBackground(QPainter *painter, const QRectF &rect)
{
    //Render scene
    std::shared_ptr<medical_imaging::SceneBase> scene = _scene.lock();
    if (!scene)
    {
        glClearColor(0.5,0.5,0.5,0);
        glClear(GL_COLOR_BUFFER_BIT );
    }
    else
    {
        //_gl_widget->makeCurrent();
        scene->initialize();
        scene->render();
        scene->render_to_back();
        //_gl_widget->doneCurrent();
    }
}

void SceneContainer::drawForeground(QPainter *painter, const QRectF &rect)
{
    //do nothing
}

void SceneContainer::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    this->setSceneRect(0,0,event->size().width() , event->size().height());

    std::shared_ptr<medical_imaging::SceneBase> scene = _scene.lock();
    if (scene)
    {
        scene->set_display_size(event->size().width() , event->size().height());
    }
}

void SceneContainer::paintEvent(QPaintEvent *event)
{
    //Update graphic item
    for (auto it = _graphic_items.begin() ; it != _graphic_items.end() ; ++it)
    {
        std::vector<QGraphicsItem*> to_be_add;
        std::vector<QGraphicsItem*> to_be_remove;
        (*it)->update(to_be_add , to_be_remove);

        //Remove
        for (auto it_remove = to_be_remove.begin() ; it_remove != to_be_remove.end() ; ++it_remove)
        {
            _inner_graphic_scene->removeItem(*it_remove);
        }

        //Add
        for (auto it_add = to_be_add.begin() ; it_add != to_be_add.end() ; ++it_add)
        {
            _inner_graphic_scene->addItem(*it_add);
        }

        //Post update
        (*it)->post_update();
    }

    QGraphicsView::paintEvent(event);
}

void SceneContainer::set_scene(std::shared_ptr<medical_imaging::SceneBase> scene)
{
    _scene = scene;
}

std::shared_ptr<medical_imaging::SceneBase> SceneContainer::get_scene()
{
    std::shared_ptr<medical_imaging::SceneBase> scene = _scene.lock();
    return scene;
}

void SceneContainer::set_name(const std::string& des)
{
    std::shared_ptr<medical_imaging::SceneBase> scene = _scene.lock();
    if (scene)
    {
        scene->set_name(des);
    }
}

std::string SceneContainer::get_name() const
{
    std::shared_ptr<medical_imaging::SceneBase> scene = _scene.lock();
    if (scene)
    {
        return scene->get_name();
    }
    else
    {
        return "";
    }
}

void SceneContainer::update_scene()
{
    _inner_graphic_scene->update();
}

void SceneContainer::slot_mouse_click()
{
    bool double_click_status = (_mouse_press_time>1);
    _mouse_press_time = 0;//Make mouse click decision
    if (double_click_status && _mouse_double_click_op)
    {
        _mouse_double_click_op->double_click(_pre_point);
    }
    update_scene();
}

void SceneContainer::mousePressEvent(QMouseEvent *event)
{
    //focus
    this->setFocus();

    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mousePressEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i())
    {
        _buttons = event->buttons();
        _pre_point = event->pos();
        _modifiers = event->modifiers();

        ++_mouse_press_time;
        if (1 == _mouse_press_time && _buttons == Qt::LeftButton) // only when left button is pressed
        {
            _buttons_pre_press = _buttons;
            //const int interval = _double_click_interval == 0 ? QApplication::doubleClickInterval() : _double_click_interval;
            //const int interval = 300;
            QTimer::singleShot( _double_click_interval , this , SLOT(slot_mouse_click()));//Use timer to decide single click and double click
        }
        else
        {
            _mouse_press_time = 0; // reset if both right and left button are pressed
        }
        //do single click directly
        IMouseOpPtrCollection ops;
        if(get_mouse_operation_i(ops))
        {
            for (auto it = ops.begin() ; it != ops.end() ; ++it)
            {
                (*it)->press(_pre_point);
            }
        }
    }
    //std::cout << "Press out >>\n";
}

void SceneContainer::mouseMoveEvent(QMouseEvent *event)
{
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mouseMoveEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i())
    {
        if (!this->get_mouse_hovering() && _buttons == Qt::NoButton)
        {
            return;
        }

        IMouseOpPtrCollection ops;
        _modifiers = event->modifiers();
        if(get_mouse_operation_i( ops))
        {
            for (auto it = ops.begin() ; it != ops.end() ; ++it)
            {
                (*it)->move(event->pos());
            }
        }

        update_scene();
    }
}

void SceneContainer::mouseReleaseEvent(QMouseEvent *event)
{
    //std::cout << "Release in >>\n";
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mouseReleaseEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i())
    {
        IMouseOpPtrCollection ops;
        _modifiers = event->modifiers();
        if(get_mouse_operation_i(ops))
        {
            for (auto it = ops.begin() ; it != ops.end() ; ++it)
            {
                (*it)->release(event->pos());
            }
        }
        _buttons = Qt::NoButton;
        update_scene();
    }
}

void SceneContainer::mouseDoubleClickEvent(QMouseEvent *event)
{
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mouseDoubleClickEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i())
    {
        if (1 == _mouse_press_time)// In timer slot to wait to run
        {
            ++_mouse_press_time;
        }
    }
}

void SceneContainer::wheelEvent(QWheelEvent *event)
{
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::wheelEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i())
    {
        if (_mouse_wheel_ops.empty())
        {
            return;
        }

        //up->positive ; down->negative
        const int degree = event->delta() / 8;//rolling angle , *8 is rolling distance
        const int step = degree/ 15;//rolling step , *15 is rolling angle

        for (auto it = _mouse_wheel_ops.begin() ; it != _mouse_wheel_ops.end() ; ++it)
        {
            (*it)->wheel_slide(step);
        }
    }
}

void SceneContainer::keyPressEvent(QKeyEvent *key)
{

}

void SceneContainer::keyReleaseEvent(QKeyEvent *key)
{

}

void SceneContainer::register_mouse_operation(std::shared_ptr<IMouseOp> mouse_op , Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier)
{
    if (mouse_op == nullptr)
    {
        _mouse_ops[button|keyboard_modifier] = IMouseOpPtrCollection(0);
    }
    else
    {
        _mouse_ops[button|keyboard_modifier] = IMouseOpPtrCollection(1 , mouse_op);
    }
}

void SceneContainer::register_mouse_operation(IMouseOpPtrCollection mouse_ops , Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier)
{
    _mouse_ops[button|keyboard_modifier] = mouse_ops;
}

bool SceneContainer::get_mouse_operation_i(IMouseOpPtrCollection& op)
{
    int key = _buttons | _modifiers;
    auto it = _mouse_ops.find(key);
    if (it != _mouse_ops.end())
    {
        op = it->second;
        return true;
    }
    else
    {
        return false;
    }
}

void SceneContainer::focusInEvent(QFocusEvent *event)
{
    emit focus_in_scene();
}

void SceneContainer::focusOutEvent(QFocusEvent *event)
{
    emit focus_out_scene();
}

IMouseOpPtrCollection SceneContainer::get_mouse_operation(Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier)
{
    int key = button | keyboard_modifier;
    auto it = _mouse_ops.find(key);
    if (it != _mouse_ops.end())
    {
        return it->second;
    }
    else
    {
        return IMouseOpPtrCollection();
    }
}

void SceneContainer::register_mouse_wheel_operation(IMouseOpPtrCollection mouse_ops)
{
    _mouse_wheel_ops = mouse_ops;
}

void SceneContainer::register_mouse_wheel_operation(IMouseOpPtr mouse_op)
{
    IMouseOpPtrCollection(1 ,mouse_op).swap(_mouse_wheel_ops);
}

void SceneContainer::register_mouse_double_click_operation(IMouseOpPtr mouse_op)
{
    _mouse_double_click_op = mouse_op;
}

bool SceneContainer::no_graphics_item_grab_i()
{
    return _inner_graphic_scene->mouseGrabberItem() == nullptr;
}

void SceneContainer::add_item(GraphicItemPtr item)
{
    _graphic_items.push_back(item);
    std::vector<QGraphicsItem*> items = item->get_init_items();
    
    for (auto it = items.begin() ; it != items.end() ; ++it)
    {
        _inner_graphic_scene->addItem(*it);
    }
}

void SceneContainer::clear()
{
    _inner_graphic_scene->clear();
    _graphic_items.clear();
    _scene.reset();
}