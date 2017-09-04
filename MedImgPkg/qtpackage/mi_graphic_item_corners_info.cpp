#include "mi_graphic_item_corners_info.h"

//Medical imaging
#include "io/mi_image_data_header.h"
#include "io/mi_image_data.h"
#include "util/mi_string_number_converter.h"
#include "renderalgo/mi_ray_cast_scene.h"
#include "renderalgo/mi_volume_infos.h"
#include "renderalgo/mi_camera_calculator.h"
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
}

GraphicItemCornersInfo::GraphicItemCornersInfo():
        _text_item_lb(new QGraphicsTextItem()),
        _text_item_lt(new QGraphicsTextItem()),
        _text_item_rb(new QGraphicsTextItem()),
        _text_item_rt(new QGraphicsTextItem()),
        _pre_ww(std::numeric_limits<int>::min()),
        _pre_wl(std::numeric_limits<int>::min()),
        _pre_window_width(-1),
        _pre_window_height(-1)
{

}

GraphicItemCornersInfo::~GraphicItemCornersInfo()
{

}

void GraphicItemCornersInfo::set_scene(std::shared_ptr<SceneBase> scene)
{
    GraphicItemBase::set_scene(scene);
    refresh_text_i();
}


std::vector<QGraphicsItem*> GraphicItemCornersInfo::get_init_items()
{
    std::vector<QGraphicsItem*> items(4);
    items[0] = _text_item_lb;
    items[1] = _text_item_lt;
    items[2] = _text_item_rb;
    items[3] = _text_item_rt;
    return std::move(items);
}

void GraphicItemCornersInfo::update(std::vector<QGraphicsItem*>& to_be_add , std::vector<QGraphicsItem*>& to_be_remove)
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene_base);
    int width , height;
    scene_base->get_display_size(width , height);

    if (_pre_window_width != width || _pre_window_height!= height)
    {
        refresh_text_i();
        return;
    }

    //update window level
    //////////////////////////////////////////////////////////////////////////
    //3.3 Left Bottom
    //Window level

    std::string context;
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
    if (mpr_scene)
    {
        //Set context
        float ww(1) , wl(0);
        mpr_scene->get_global_window_level(ww , wl);
        int wl_int = (int)wl;
        int ww_int = (int)ww;
        if (wl_int == _pre_wl && ww_int == _pre_ww)
        {
            return;
        }
        else
        {
            _pre_wl = wl_int;
            _pre_ww = ww_int;
        }
        

        StrNumConverter<int> num_to_str;
        context = std::string("C : ") + num_to_str.to_string(wl_int) + std::string("  W : ") + num_to_str.to_string(ww_int); 

        //Set alignment
        QTextDocument* dcm = _text_item_lb->document();
        dcm->clear();
        dcm->setPlainText(context.c_str());
        QTextOption option = dcm->defaultTextOption();
        option.setAlignment(Qt::AlignLeft);
        dcm->setDefaultTextOption(option);
        //_text_item_lb->setTextWidth(dcm->idealWidth());

        //Set position
        _text_item_lb->setPos(BORDER , height - (BORDER+MARGIN*2));
    }

}

void GraphicItemCornersInfo::refresh_text_i()
{
    std::shared_ptr<SceneBase> scene_base = _scene.lock();
    QTWIDGETS_CHECK_NULL_EXCEPTION(scene_base);

    std::shared_ptr<RayCastScene> ray_cast_scene = std::dynamic_pointer_cast<RayCastScene>(scene_base);
    QTWIDGETS_CHECK_NULL_EXCEPTION(ray_cast_scene);

    std::shared_ptr<VolumeInfos> volume_infos = ray_cast_scene->get_volume_infos();
    QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);

    std::shared_ptr<ImageDataHeader> data_header = volume_infos->get_data_header();
    QTWIDGETS_CHECK_NULL_EXCEPTION(data_header );

    int width(1),height(1);
    ray_cast_scene->get_display_size(width , height);

    //set font 
    QFont font("Times" , POINT_SIZE , QFont::Bold);

    _text_item_lb->setFont(font);
    _text_item_lb->setDefaultTextColor(QColor(220,220,220));

    _text_item_lt->setFont(font);
    _text_item_lt->setDefaultTextColor(QColor(220,220,220));

    _text_item_rb->setFont(font);
    _text_item_rb->setDefaultTextColor(QColor(220,220,220));

    _text_item_rt->setFont(font);
    _text_item_rt->setDefaultTextColor(QColor(220,220,220));

    //////////////////////////////////////////////////////////////////////////
    //Patient four corners info
    //////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////
    //3.1 Left Top
    //Manufacturer
    //Manufacturer model name
    //Modality
    //Image date
    std::string modality = "UnSupported";
    switch(data_header->modality)
    {
    case CR:
        modality = "CR";
        break;
    case CT:
        modality = "CT";
        break;
    case MR:
        modality = "MR";
        break;
    case PT:
        modality = "PT";
        break;
    default:
        break;
    }

    //Set context
    std::string context = data_header->manufacturer + CRLF +
        data_header->manufacturer_model_name + CRLF +
        modality + CRLF +
        data_header->image_date + CRLF;

    //Set alignment
    QTextDocument* dcm = _text_item_lt->document();
    dcm->setPlainText(context.c_str());
    QTextOption option1 = dcm->defaultTextOption();
    option1.setAlignment(Qt::AlignLeft);
    dcm->setDefaultTextOption(option1);
    _text_item_lt->setTextWidth(dcm->idealWidth());

    //Set position
    _text_item_lt->setPos(BORDER,BORDER);

    //////////////////////////////////////////////////////////////////////////
    //3.2 Right Top
    //Patient name
    //Patient ID
    //Patient Sex
    //Image dimension

    //Set context
    std::string().swap(context);
    context = data_header->patient_name + CRLF +
        data_header->patient_id + CRLF;
    if (!data_header->patient_sex.empty())
    {
        context += data_header->patient_sex + CRLF;
    }

    {
        std::stringstream ss;
        ss << data_header->columns << " " << data_header->rows << " " << data_header->slice_location.size();
        context += ss.str();
    }

    //Set alignment
    dcm = _text_item_rt->document();
    dcm->setPlainText(context.c_str());
    QTextOption option2 = dcm->defaultTextOption();
    option2.setAlignment(Qt::AlignRight);
    dcm->setDefaultTextOption(option2);
    _text_item_rt->setTextWidth(dcm->idealWidth());

    //Set position
    _text_item_rt->setPos(width - (BORDER+_text_item_rt->textWidth()),BORDER);

    //////////////////////////////////////////////////////////////////////////
    //3.4 Right bottom
    //KVP
    //Thickness
    std::string().swap(context);
    {
        std::stringstream ss;
        ss << "kvp:" << data_header->kvp << CRLF;
        context = ss.str();
    }

    double thickness = 0.0;
    //if (scene_base->get_camera()->get_eye() != Point3(0.0, 0.0, 0.0) )
    //{
        Vector3 view_to = scene_base->get_camera()->get_view_direction();
        std::shared_ptr<CameraCalculator> camera_cal = ray_cast_scene->get_camera_calculator();
        if (  (1.0 - abs(view_to.dot_product(Vector3(1.0, 0.0, 0.0)))) < 1e-6 )
        {
            thickness = volume_infos->get_volume()->_spacing[camera_cal->get_left_patient_axis_info().volume_coord / 2]; 
        }
        else if ((1.0 - abs(view_to.dot_product(Vector3(0.0, 1.0, 0.0)))) < 1e-6)
        {
            thickness = volume_infos->get_volume()->_spacing[camera_cal->get_posterior_patient_axis_info().volume_coord / 2]; 
        }
        else if ((1.0 - abs(view_to.dot_product(Vector3(0.0, 0.0, 1.0)))) < 1e-6 )
        {
            thickness = volume_infos->get_volume()->_spacing[camera_cal->get_head_patient_axis_info().volume_coord / 2];
        }
    //}
    {
        std::stringstream ss;
        ss << "thickness:" << thickness;
        context += ss.str();
    }

    dcm = _text_item_rb->document();
    dcm->setPlainText(context.c_str());
    QTextOption option4 = dcm->defaultTextOption();
    option4.setAlignment(Qt::AlignRight);
    dcm->setDefaultTextOption(option4);
    dcm->adjustSize();
    _text_item_rb->setTextWidth(dcm->idealWidth());
    std::cout << _text_item_rb->textWidth() << std::endl;
    //Set position
    _text_item_rb->setPos(width - (BORDER+_text_item_rb->textWidth()), height - (BORDER+MARGIN*4));

    //////////////////////////////////////////////////////////////////////////
    //3.3 Left Bottom
    //Window level

    std::string().swap(context);
    std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(scene_base);
    if (mpr_scene)
    {
        //Set context
        float ww(1) , wl(0);
        mpr_scene->get_global_window_level(ww , wl);
        int wl_int = (int)wl;
        int ww_int = (int)ww;

        StrNumConverter<int> num_to_str;
        context = std::string("C : ") + num_to_str.to_string(wl_int) + std::string("  W : ") + num_to_str.to_string(ww_int); 

        //Set alignment
        dcm = _text_item_lb->document();
        dcm->setPlainText(context.c_str());
        QTextOption option3 = dcm->defaultTextOption();
        option3.setAlignment(Qt::AlignLeft);
        dcm->setDefaultTextOption(option3);
        //_text_item_lb->setTextWidth(dcm->idealWidth());

        //Set position
        _text_item_lb->setPos(BORDER , height - (BORDER+MARGIN*2));

        _pre_ww = ww_int;
        _pre_wl = wl_int;
    }

    
    _pre_window_width = width;
    _pre_window_height = height;
}



MED_IMG_END_NAMESPACE