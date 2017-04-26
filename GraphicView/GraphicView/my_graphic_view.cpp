#include "my_graphic_view.h"

#include <QResizeEvent>

#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgQtWidgets/mi_shared_widget.h"

#include "my_graphic_scene.h"

using namespace medical_imaging;

MyGraphicView::MyGraphicView(QWidget *parent ):
        QGraphicsView(parent)
{
    _graphic_scene = new MyGraphicScene();
    this->setScene(_graphic_scene);

    _gl_widget = new QGLWidget(0 , SharedWidget::instance() );
    this->setViewport(_gl_widget );

    //this->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
}

MyGraphicView::~MyGraphicView()
{

}

void MyGraphicView::drawBackground(QPainter *painter, const QRectF &rect)
{
    static int idx = 0;
    std::cout << "draw background " << idx++ << std::endl;
    if (!_scene)
    {
        glClearColor(0.5,0.5,0.5,0);
        glClear(GL_COLOR_BUFFER_BIT );
    }
    else
    {
        //_gl_widget->makeCurrent();

        _scene->initialize();
        _scene->render(0);
        _scene->render_to_back();

        //_gl_widget->doneCurrent();
    }
}

void MyGraphicView::drawForeground(QPainter *painter, const QRectF &rect)
{
    //do nothing
}

void MyGraphicView::set_scene(std::shared_ptr<SceneBase> scene)
{
    _scene = scene;
}

void MyGraphicView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    this->setSceneRect(0,0,event->size().width() , event->size().height());

    if (_scene)
    {
        _scene->set_display_size(event->size().width() , event->size().height());
    }
}

void MyGraphicView::update_scene()
{
    _graphic_scene->update();
}

MyGraphicScene* MyGraphicView::get_graphics_scene()
{
    return _graphic_scene;
}

void MyGraphicView::mousePressEvent(QMouseEvent *event)
{
    QGraphicsView::mousePressEvent(event);

    if (_graphic_scene->mouseGrabberItem() == nullptr)
    {
        std::cout << "View press\n";
    }
    else
    {
        std::cout << "View no press\n";
    }
}

void MyGraphicView::mouseMoveEvent(QMouseEvent *event)
{
    QGraphicsView::mouseMoveEvent(event);

    if (_graphic_scene->mouseGrabberItem() == nullptr)
    {
        std::cout << "View move\n";
    }
    else
    {
        std::cout << "View no move\n";
    }
}
