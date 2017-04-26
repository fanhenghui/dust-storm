#include "my_graphic_view.h"

#include <QResizeEvent>

#include "MedImgRenderAlgorithm/mi_scene_base.h"
#include "MedImgQtWidgets/mi_shared_widget.h"

//#include "my_gl_widget.h"

using namespace medical_imaging;

MyGraphicView::MyGraphicView(QGraphicsScene *scene, QWidget *parent ):
        QGraphicsView(scene, parent),_gl_widget(nullptr)
{
    _gl_widget = new QGLWidget(0 , SharedWidget::instance() );
    this->setViewport(_gl_widget );
    //this->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
}

MyGraphicView::~MyGraphicView()
{

}

void MyGraphicView::drawBackground(QPainter *painter, const QRectF &rect)
{
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
