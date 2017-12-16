#include "mi_graphic_item_direction_info.h"

//Medical imaging
#include "io/mi_image_data_header.h"
#include "util/mi_string_number_converter.h"
#include "renderalgo/mi_ray_cast_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_mpr_scene.h"

#include <QGraphicsTextItem>
#include <QFont>
#include <QTextDocument>
#include <QTextOption>

#include <string>

MED_IMG_BEGIN_NAMESPACE

namespace
{
    const int POINT_SIZE = 12;
    const int MARGIN = POINT_SIZE+2;
    const std::string CRLF("\n");
    const int BORDER = 2;
    
    const Vector3 coordinate[6] = { 
        Vector3(1.0, 0.0, 0.0), 
        Vector3(-1.0, 0.0, 0.0),
        Vector3(0.0, 0.0, 1.0),
        Vector3(0.0, 0.0, -1.0),
        Vector3(0.0, 1.0, 0.0),
        Vector3(0.0, -1.0, 0.0)};
        const std::string coordinateLable[6] = {
            "L", "R", "H", "F", "P", "A"};
        //const std::string coordinateLable[6] = {
        //    "L", "R", "Head", "Foot", "Back", "Front"};
}

GraphicItemDirectionInfo::GraphicItemDirectionInfo():
        _text_item_left(new QGraphicsTextItem()),
        _text_item_right(new QGraphicsTextItem()),
        _text_item_top(new QGraphicsTextItem()),
        _text_item_bottom(new QGraphicsTextItem()),
        _pre_window_width(-1),
        _pre_window_height(-1)
{
}

GraphicItemDirectionInfo::~GraphicItemDirectionInfo()
{

}

void GraphicItemDirectionInfo::set_scene(std::shared_ptr<SceneBase> scene)
{
    GraphicItemBase::set_scene(scene);
    refresh_text();
}


std::vector<QGraphicsItem*> GraphicItemDirectionInfo::get_init_items()
{
    std::vector<QGraphicsItem*> items(4);
    items[0] = _text_item_left;
    items[1] = _text_item_right;
    items[2] = _text_item_top;
    items[3] = _text_item_bottom;
    return std::move(items);
}

void GraphicItemDirectionInfo::update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene_base);
    int width , height;
    scene_base->get_display_size(width , height);
    Vector3 view_to = scene_base->get_camera()->get_view_direction();
    view_to *= -1.0;
    Vector3 view_up = scene_base->get_camera()->get_up_direction();
    Vector3 view_left = view_up.cross_product(view_to);

    if (_pre_window_width != width || _pre_window_height != height || 
        view_left != this->_slice_left || view_up != this->_slice_up)
    {
        _pre_window_width = width;
        _pre_window_height = height;
        this->_slice_left = view_left;
        this->_slice_up = view_up;
        refresh_text();
        return;
    }
}

void GraphicItemDirectionInfo::refresh_text()
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene_base);

    std::shared_ptr<RayCastScene> ray_cast_scene = std::dynamic_pointer_cast<RayCastScene>(scene_base);
    QTWIDGETS_CHECK_NULL_EXCEPTION(ray_cast_scene);

    std::shared_ptr<VolumeInfos> volume_infos = ray_cast_scene->get_volume_infos();
    QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);

    //set font 
    QFont font("Times" , POINT_SIZE , QFont::Bold);

    _text_item_left->setFont(font);
    _text_item_left->setDefaultTextColor(QColor(220,220,220));

    _text_item_right->setFont(font);
    _text_item_right->setDefaultTextColor(QColor(220,220,220));

    _text_item_top->setFont(font);
    _text_item_top->setDefaultTextColor(QColor(220,220,220));

    _text_item_bottom->setFont(font);
    _text_item_bottom->setDefaultTextColor(QColor(220,220,220));

    // find the axis with smallest angle with slice_left
    for (int i=0; i<6; ++i)
    {
        if( (1.0 - this->_slice_left.dot_product(coordinate[i])) < 1e-6 )
        {
            QTextDocument* dcm = this->_text_item_left->document();
            dcm->setPlainText(coordinateLable[i].c_str());
            this->_text_item_left->setTextWidth(dcm->idealWidth());
            this->_text_item_left->setPos(_pre_window_width-_text_item_left->textWidth()-BORDER, _pre_window_height/2.0);
            
            int j = i%2==0 ? i+1 : i-1;
            dcm = this->_text_item_right->document();
            dcm->setPlainText(coordinateLable[j].c_str());
            this->_text_item_right->setTextWidth(dcm->idealWidth());
            this->_text_item_right->setPos(BORDER, _pre_window_height/2.0);

            break;
        }
    }
    // find the axis with smallest angle with slice_up
    for (int i=0; i<6; ++i)
    {
        if( (1.0 - this->_slice_up.dot_product(coordinate[i])) < 1e-6 )
        {
            QTextDocument* dcm = this->_text_item_top->document();
            dcm->setPlainText(coordinateLable[i].c_str());
            this->_text_item_top->setTextWidth(dcm->idealWidth());
            this->_text_item_top->setPos(_pre_window_width/2.0 - this->_text_item_top->textWidth()/2.0, BORDER);

            int j = i%2==0 ? i+1 : i-1;
            dcm = this->_text_item_bottom->document();
            dcm->setPlainText(coordinateLable[j].c_str());
            this->_text_item_bottom->setTextWidth(dcm->idealWidth());
            this->_text_item_bottom->setPos(_pre_window_width/2.0 - this->_text_item_bottom->textWidth()/2.0, _pre_window_height-(BORDER+MARGIN*2) );
            break;
        }
    }
}



MED_IMG_END_NAMESPACE