#include "mi_scene_container.h"

#include <QMouseEvent>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QTimer>
#include <QApplication>

#include "renderalgo/mi_scene_base.h"

#include "mi_shared_widget.h"
#include "mi_mouse_op_interface.h"
#include "mi_graphic_item_base.h"

using namespace medical_imaging;


Graphics2DScene::Graphics2DScene(QObject* parent /*= 0*/):
    QGraphicsScene(parent) {

}

Graphics2DScene::~Graphics2DScene() {

}

//////////////////////////////////////////////////////////////////////////

SceneContainer::SceneContainer(QWidget* parent /*= 0*/):
    QGraphicsView(parent),
    _buttons(Qt::NoButton),
    _modifiers(Qt::NoModifier),
    _mouse_press_time(0),
    _mouse_release_time(0),
    _pre_point(0, 0),
    _buttons_pre_press(Qt::NoButton) {
    _inner_graphic_scene = new Graphics2DScene();
    this->setScene(_inner_graphic_scene);

    _inner_gl_widget = new QGLWidget(0 , SharedWidget::instance());
    this->setViewport(_inner_gl_widget);

    //this->setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
}

SceneContainer::~SceneContainer() {

}

void SceneContainer::drawBackground(QPainter* painter, const QRectF& rect) {
    //static int idx = 0;
    //std::cout << "draw background " << idx++ << std::endl;

    //Render scene
    if (!_scene) {
        glClearColor(0.5, 0.5, 0.5, 0);
        glClear(GL_COLOR_BUFFER_BIT);
    } else {
        //_gl_widget->makeCurrent();

        _scene->initialize();
        _scene->render(0);
        _scene->render_to_back();

        //_gl_widget->doneCurrent();
    }
}

void SceneContainer::drawForeground(QPainter* painter, const QRectF& rect) {
    //do nothing
}

void SceneContainer::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    this->setSceneRect(0, 0, event->size().width() , event->size().height());

    if (_scene) {
        _scene->set_display_size(event->size().width() , event->size().height());
    }
}

void SceneContainer::paintEvent(QPaintEvent* event) {
    //Update graphic item
    for (auto it = _graphic_items.begin() ; it != _graphic_items.end() ; ++it) {
        std::vector<QGraphicsItem*> to_be_add;
        std::vector<QGraphicsItem*> to_be_remove;
        (*it)->update(to_be_add , to_be_remove);

        //Remove
        for (auto it_remove = to_be_remove.begin() ; it_remove != to_be_remove.end() ; ++it_remove) {
            _inner_graphic_scene->removeItem(*it_remove);
        }

        //Add
        for (auto it_add = to_be_add.begin() ; it_add != to_be_add.end() ; ++it_add) {
            _inner_graphic_scene->addItem(*it_add);
        }

        //Post update
        (*it)->post_update();
    }

    QGraphicsView::paintEvent(event);
}

void SceneContainer::set_scene(std::shared_ptr<medical_imaging::SceneBase> scene) {
    _scene = scene;
}

std::shared_ptr<medical_imaging::SceneBase> SceneContainer::get_scene() {
    return _scene;
}

void SceneContainer::set_name(const std::string& des) {
    if (_scene) {
        _scene->set_name(des);
    }
}

std::string SceneContainer::get_name() const {
    if (_scene) {
        return _scene->get_name();
    } else {
        return "";
    }
}

void SceneContainer::update_scene() {
    _inner_graphic_scene->update();
    //std::cout << "update scene" << std::endl;
}

void SceneContainer::slot_mouse_click() {
    //std::cout << "Slot click in >>\n";
    bool double_click_status = (_mouse_press_time > 1) && (_buttons_pre_press == _buttons);
    _mouse_press_time = 0;//Make mouse click decision

    if (!double_click_status) {
        //std::cout << "Run single click in slot \n";
        IMouseOpPtrCollection ops;

        if (get_mouse_operation_i(ops)) {
            for (auto it = ops.begin() ; it != ops.end() ; ++it) {
                (*it)->press(_pre_point);
            }
        }
    } else {
        //std::cout << "Run double click in slot \n";
        IMouseOpPtrCollection ops;

        if (get_mouse_operation_i(ops)) {
            for (auto it = ops.begin() ; it != ops.end() ; ++it) {
                (*it)->double_click(_pre_point);
            }
        }
    }

    //Do previous release logic when decide the click type
    if (1 == _mouse_release_time) {
        //std::cout << "Mouse release \n";
        IMouseOpPtrCollection ops;

        if (get_mouse_operation_i(ops)) {
            for (auto it = ops.begin() ; it != ops.end() ; ++it) {
                (*it)->release(_pre_point);
            }
        }

        _buttons = Qt::NoButton;
        _mouse_release_time = 0;
        //std::cout << "release in slot\n";
    }

    update_scene();

    //std::cout << "Slot click out <<\n";
}

void SceneContainer::mousePressEvent(QMouseEvent* event) {
    //std::cout << "\nPress in >>\n";
    //focus
    this->setFocus();

    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mousePressEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i()) {
        //std::cout << "No graphics grab \n";
        _buttons = event->buttons();
        _pre_point = event->pos();
        _modifiers = event->modifiers();

        //std::cout << "Previous mouse press time : " << _mouse_press_time << std::endl;
        ++_mouse_press_time;

        if (1 == _mouse_press_time) {
            //std::cout << "Trigger timer\n";
            _buttons_pre_press = _buttons;
            //const int interval = QApplication::doubleClickInterval();
            const int interval = 150;
            QTimer::singleShot(interval , this ,
                               SLOT(slot_mouse_click())); //Use timer to decide single click and double click
        }
    }

    //std::cout << "Press out >>\n";
}

void SceneContainer::mouseMoveEvent(QMouseEvent* event) {
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mouseMoveEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i()) {
        if (_buttons == Qt::NoButton) {
            return;
        }

        if (0 == _mouse_press_time) { //mouse after press
            IMouseOpPtrCollection ops;
            _modifiers = event->modifiers();

            if (get_mouse_operation_i(ops)) {
                for (auto it = ops.begin() ; it != ops.end() ; ++it) {
                    (*it)->move(event->pos());
                }
            }

            update_scene();
        }
    }
}

void SceneContainer::mouseReleaseEvent(QMouseEvent* event) {
    //std::cout << "Release in >>\n";
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mouseReleaseEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i()) {
        //std::cout << "No graphics grab \n";

        if (0 == _mouse_press_time) {
            _mouse_release_time = 0;

            IMouseOpPtrCollection ops;
            _modifiers = event->modifiers();

            if (get_mouse_operation_i(ops)) {
                for (auto it = ops.begin() ; it != ops.end() ; ++it) {
                    (*it)->release(event->pos());
                }
            }

            _buttons = Qt::NoButton;
            update_scene();
        } else {
            _mouse_release_time = 1;
        }
    }

    //std::cout << "Release out <<\n";
}

void SceneContainer::mouseDoubleClickEvent(QMouseEvent* event) {
    //std::cout << "Double click in >>\n";
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::mouseDoubleClickEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i()) {
        //std::cout << "No graphics grab \n";
        if (1 == _mouse_press_time) { // In timer slot to wait to run
            ++_mouse_press_time;
            _buttons = event->buttons();
        } else {
            //std::cout << "Double click to be late......\n";
        }
    }

    //std::cout << "Double click out <<\n";
}

void SceneContainer::wheelEvent(QWheelEvent* event) {
    //1 Graphic item(Qt 2D) interaction
    QGraphicsView::wheelEvent(event);

    //2 TODO Graphic primitive(3D) interaction

    //3 Mouse operation
    if (no_graphics_item_grab_i()) {
        if (_mouse_wheel_ops.empty()) {
            return;
        }

        //���������Ǹ���������������
        const int degree = event->delta() /
                           8;//�����ĽǶȣ�*8�����������ľ���
        const int step = degree /
                         15; //�����Ĳ�����*15�����������ĽǶ�

        for (auto it = _mouse_wheel_ops.begin() ; it != _mouse_wheel_ops.end() ; ++it) {
            (*it)->wheel_slide(step);
        }
    }
}

void SceneContainer::keyPressEvent(QKeyEvent* key) {

}

void SceneContainer::keyReleaseEvent(QKeyEvent* key) {

}

void SceneContainer::register_mouse_operation(std::shared_ptr<IMouseOp> mouse_op ,
        Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier) {
    _mouse_ops[button | keyboard_modifier] = IMouseOpPtrCollection(1 , mouse_op);
}

void SceneContainer::register_mouse_operation(IMouseOpPtrCollection mouse_ops ,
        Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier) {
    _mouse_ops[button | keyboard_modifier] = mouse_ops;
}

bool SceneContainer::get_mouse_operation_i(IMouseOpPtrCollection& op) {
    int key = _buttons | _modifiers;
    auto it = _mouse_ops.find(key);

    if (it != _mouse_ops.end()) {
        op = it->second;
        return true;
    } else {
        return false;
    }
}

void SceneContainer::focusInEvent(QFocusEvent* event) {
    emit focus_in_scene();
}

void SceneContainer::focusOutEvent(QFocusEvent* event) {
    emit focus_out_scene();
}

IMouseOpPtrCollection SceneContainer::get_mouse_operation(Qt::MouseButtons button ,
        Qt::KeyboardModifier keyboard_modifier) {
    int key = button | keyboard_modifier;
    auto it = _mouse_ops.find(key);

    if (it != _mouse_ops.end()) {
        return it->second;
    } else {
        return IMouseOpPtrCollection();
    }
}

void SceneContainer::register_mouse_wheel_operation(IMouseOpPtrCollection mouse_ops) {
    _mouse_wheel_ops = mouse_ops;
}

void SceneContainer::register_mouse_wheel_operation(IMouseOpPtr mouse_op) {
    IMouseOpPtrCollection(1 , mouse_op).swap(_mouse_wheel_ops);
}

bool SceneContainer::no_graphics_item_grab_i() {
    return _inner_graphic_scene->mouseGrabberItem() == nullptr;
}

void SceneContainer::add_item(GraphicItemPtr item) {
    _graphic_items.push_back(item);
    std::vector<QGraphicsItem*> items = item->get_init_items();

    for (auto it = items.begin() ; it != items.end() ; ++it) {
        _inner_graphic_scene->addItem(*it);
    }
}

void SceneContainer::clear() {
    _inner_graphic_scene->clear();
    _graphic_items.clear();
    _scene.reset();
}

//////////////////////////////////////////////////////////////////////////
