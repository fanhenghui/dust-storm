#ifndef MYGRAPHICVIEW_H
#define MYGRAPHICVIEW_H

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>

namespace medical_imaging
{
    class SceneBase;
}

//class MyGLWidget;

class MyGraphicView : public QGraphicsView
{
    Q_OBJECT

public:
    MyGraphicView(QGraphicsScene *scene, QWidget *parent = 0);
    ~MyGraphicView();

    void set_scene(std::shared_ptr<medical_imaging::SceneBase> scene);

protected:
    void resizeEvent(QResizeEvent *event);
    virtual void drawBackground(QPainter *painter, const QRectF &rect);
    virtual void drawForeground(QPainter *painter, const QRectF &rect);

private:
    //MyGLWidget* _my_widget;

    std::shared_ptr<medical_imaging::SceneBase> _scene;

    QGLWidget* _gl_widget;
};

#endif // MYGRAPHICVIEW_H
