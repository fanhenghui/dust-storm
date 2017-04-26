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

class MyGraphicScene;
class MyGraphicView : public QGraphicsView
{
    Q_OBJECT

public:
    MyGraphicView(QWidget *parent = 0);
    ~MyGraphicView();

    void set_scene(std::shared_ptr<medical_imaging::SceneBase> scene);

    void update_scene();

    MyGraphicScene* get_graphics_scene();


protected:
    void resizeEvent(QResizeEvent *event);
    virtual void drawBackground(QPainter *painter, const QRectF &rect);
    virtual void drawForeground(QPainter *painter, const QRectF &rect);

    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);

private:
    //MyGLWidget* _my_widget;

    std::shared_ptr<medical_imaging::SceneBase> _scene;

    QGLWidget* _gl_widget;
    MyGraphicScene* _graphic_scene;
};

#endif // MYGRAPHICVIEW_H
