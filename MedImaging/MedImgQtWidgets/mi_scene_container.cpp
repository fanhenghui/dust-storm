#include "mi_scene_container.h"

#include <QMouseEvent>
#include <QPainter>
#include <QTimer>

#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgRenderAlgorithm/mi_scene_base.h"

#include "mi_shared_widget.h"
#include "mi_painter_base.h"
#include "mi_mouse_op_interface.h"

using namespace medical_imaging;

SceneContainer::SceneContainer(SharedWidget* shared , QWidget* parent /*= 0*/):
        QGLWidget(parent , shared),_buttons(Qt::NoButton),_modifiers(Qt::NoModifier),_pixel_map(new QPixmap(256,256)),_mouse_press_time(0),_mouse_release_time(0)
{
    makeCurrent();

    setAutoBufferSwap( false );
    setAutoFillBackground( false );
    //setMouseTracking(true);
}

SceneContainer::~SceneContainer()
{
    makeCurrent();

    /*if (_scene)
    {
        _scene->finalize();
    }*/
}

void SceneContainer::initializeGL()
{
    if (GLEW_OK != glewInit())
    {
        QTWIDGETS_THROW_EXCEPTION("Glew init failed!");
    }

    if (_scene)
    {
        _scene->initialize();
    }
}

void SceneContainer::paintEvent(QPaintEvent* painter)
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    if (!_scene)
    {
        makeCurrent();

        glViewport(0,0,this->width() , this->height());
        glFrontFace( GL_CW );
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glClearColor(0,0,0,0);
        glClear(GL_COLOR_BUFFER_BIT);

        QPainter painter(this);
        painter.setPen(Qt::red);
        painter.drawText(this->width()/2,this->height()/2,tr("NULL"));
        swapBuffers();

        return;
    }
    else
    {
        makeCurrent();

        CHECK_GL_ERROR;
        if (_scene)
        {
            _scene->initialize();
        }
        CHECK_GL_ERROR;
        //1 Scene rendering
        _scene->render(0);
        _scene->render_to_back();

        //2 Painter drawing
        _pixel_map->fill(Qt::transparent);
        QPainter painter(_pixel_map.get());
        for (auto it = _painters.begin() ; it != _painters.end()  ;++it)
        {
            (*it)->set_painter(&painter);
            (*it)->render();
        }

        QPainter p(this);
        p.drawPixmap(0,0,*_pixel_map);


        //3 Swap buffers
        swapBuffers( );
    }
}

void SceneContainer::slot_mouse_click()
{
    bool double_click_status = (_mouse_press_time>1) && (_buttons_pre_press == _buttons);
    if (!double_click_status)
    {
        //std::cout << "single click in slot \n";
        IMouseOpPtrCollection ops;
        if(get_mouse_operation_i(ops))
        {
            for (auto it = ops.begin() ; it != ops.end() ; ++it)
            {
                (*it)->press(_pre_point);
            }
        }
    }
    else
    {
        //std::cout << "double click in slot \n";
        IMouseOpPtrCollection ops;
        if(get_mouse_operation_i( ops))
        {
            for (auto it = ops.begin() ; it != ops.end() ; ++it)
            {
                (*it)->double_click(_pre_point);
            }
        }
    }

    //Do previous release logic when decide the click type
    if (1 == _mouse_release_time)
    {
        IMouseOpPtrCollection ops;
        if(get_mouse_operation_i(ops))
        {
            for (auto it = ops.begin() ; it != ops.end() ; ++it)
            {
                (*it)->release(_pre_point);
            }
        }

        _buttons = Qt::NoButton;
        _mouse_release_time = 0;
        //std::cout << "release in slot\n";
    }

    _mouse_press_time = 0;
    update();
}

void SceneContainer::mousePressEvent(QMouseEvent *event)
{
    //focus
    this->setFocus();

    _buttons = event->buttons();
    _pre_point = event->pos();
    _modifiers = event->modifiers();

    ++_mouse_press_time;
    if (1 == _mouse_press_time)
    {
        _buttons_pre_press = _buttons;
        QTimer::singleShot(200 , this , SLOT(slot_mouse_click()));//Use timer to decide single click and double click
    }
}

void SceneContainer::mouseReleaseEvent(QMouseEvent *event)
{
    if (0 == _mouse_press_time)
    {
        _mouse_release_time = 0;

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
        update();
    }
    else
    {
        _mouse_release_time = 1;
    }
}

void SceneContainer::mouseMoveEvent(QMouseEvent *event)
{
    if (0 == _mouse_press_time)//mouse after press 
    {
        IMouseOpPtrCollection ops;
        _modifiers = event->modifiers();
        if(get_mouse_operation_i( ops))
        {
            for (auto it = ops.begin() ; it != ops.end() ; ++it)
            {
                (*it)->move(event->pos());
            }
        }

        update();
    }
}

void SceneContainer::mouseDoubleClickEvent(QMouseEvent *event)
{
    ++_mouse_press_time;
    _buttons = event->buttons();
}

void SceneContainer::wheelEvent(QWheelEvent *event)
{
    if (_mouse_wheel_ops.empty())
    {
        return;
    }

    //滚轮向下是负数，向上是正数
    const int degree = event->delta() / 8;//滚动的角度，*8就是鼠标滚动的距离
    const int step = degree/ 15;//滚动的步数，*15就是鼠标滚动的角度

    for (auto it = _mouse_wheel_ops.begin() ; it != _mouse_wheel_ops.end() ; ++it)
    {
        (*it)->wheel_slide(step);
    }
}

void SceneContainer::keyPressEvent(QKeyEvent *key)
{
    
}

void SceneContainer::keyReleaseEvent(QKeyEvent *key)
{

}

void SceneContainer::resizeGL(int w, int h)
{
    boost::unique_lock<boost::mutex> locker(_mutex);
    if (_scene)
    {
        _scene->set_display_size(w,h);
    }
    _pixel_map.reset(new QPixmap(w,h));
}

void SceneContainer::set_scene(std::shared_ptr<medical_imaging::SceneBase> scene)
{
    _scene = scene;
}

void SceneContainer::add_painter_list(std::vector<std::shared_ptr<PainterBase>> painters)
{
    _painters = painters;
}

void SceneContainer::register_mouse_operation(std::shared_ptr<IMouseOp> mouse_op , Qt::MouseButtons button , Qt::KeyboardModifier keyboard_modifier)
{
    _mouse_ops[button|keyboard_modifier] = IMouseOpPtrCollection(1 , mouse_op);
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

void SceneContainer::set_name(const std::string& des)
{
    if (_scene)
    {
        _scene->set_name(des);
    }
}

std::string SceneContainer::get_name() const
{
    if (_scene)
    {
        return _scene->get_name();
    }
    else
    {
        return "";
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
