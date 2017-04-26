#include "my_graphic_scene.h"

#include <iostream>
#include <QGraphicsSceneMouseEvent>


MyGraphicScene::MyGraphicScene(QObject* parent /*= 0*/):_no_item_selected(false)
{

}

MyGraphicScene::~MyGraphicScene()
{

}

void MyGraphicScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    std::cout << "mouse press\n";

    QGraphicsScene::mousePressEvent(event);

    _no_item_selected = (mouseGrabberItem() == nullptr);

    //if (this->selectedItems().empty())
    //{
    //    std::cout << "no selected\n";
    //}
}

void MyGraphicScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    //std::cout << "mouse move\n";

    QGraphicsScene::mouseMoveEvent(event);

    /*if (this->selectedItems().empty())
    {
        std::cout << "no selected\n";
    }*/
}

void MyGraphicScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    //std::cout << "mouse release\n";
    QGraphicsScene::mouseReleaseEvent(event);
}

void MyGraphicScene::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    std::cout << "mouse double click\n";
}

void MyGraphicScene::wheelEvent(QGraphicsSceneWheelEvent *event)
{
    std::cout << "mouse wheel\n";
}

bool MyGraphicScene::no_item_selected()
{
    return _no_item_selected;
}
