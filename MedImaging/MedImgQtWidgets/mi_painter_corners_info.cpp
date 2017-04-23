#include "mi_painter_corners_info.h"

//Medical imaging
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgCommon/mi_string_number_converter.h"
#include "MedImgRenderAlgorithm/mi_ray_cast_scene.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

//Qt
#include "qt/qobject.h"
#include "qt/qpainter.h"
#include "qt/qstring.h"
#include "qt/qpainter.h"
#include "qt/qlabel.h"

MED_IMAGING_BEGIN_NAMESPACE

CornersInfoPainter::CornersInfoPainter()
{

}

CornersInfoPainter::~CornersInfoPainter()
{

}

void CornersInfoPainter::render()//TODO patient four corners info by configuration way
{
    try
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(_scene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(_painter);

        std::shared_ptr<RayCastScene> scene = std::dynamic_pointer_cast<RayCastScene>(_scene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(scene);

        std::shared_ptr<VolumeInfos> volume_infos = scene->get_volume_infos();
        QTWIDGETS_CHECK_NULL_EXCEPTION(volume_infos);

        std::shared_ptr<ImageDataHeader> data_header = volume_infos->get_data_header();
        QTWIDGETS_CHECK_NULL_EXCEPTION(data_header );

        int width(1),height(1);
        scene->get_display_size(width , height);

        //1 Set font
        const int point_size = 12;
        const int margin = point_size+2;
        const int border = 3;
        QFont font("Times" , point_size , QFont::Bold);
        _painter->setFont(font);

        //2 Set color
        _painter->setPen(QColor(220,220,220));

        //////////////////////////////////////////////////////////////////////////
        //3 Patient four corners info
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //3.1 Left Top
        int iItem = 1;
        //3.1.1
        _painter->drawText(border , (iItem++)*margin , data_header->manufacturer.c_str());
        //3.1.2
        _painter->drawText(border , (iItem++)*margin , data_header->manufacturer_model_name.c_str());
        //3.1.3
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
        _painter->drawText(border , (iItem++)*margin , modality.c_str());
        //3.1.4
        _painter->drawText(border , (iItem++)*margin , data_header->image_date.c_str());

        //////////////////////////////////////////////////////////////////////////
        //3.2 Right Top
        iItem = 1;
        int x = 0;
        //3.2.1
        QString patient_name(data_header->patient_name.c_str());
        x = width - _painter->fontMetrics().width(patient_name) - border;
        _painter->drawText(x , (iItem++)*margin , patient_name);

        //3.2.2
        QString patient_id(data_header->patient_id.c_str());
        x = width - _painter->fontMetrics().width(patient_id) - border;
        _painter->drawText(x , (iItem++)*margin , patient_id);

        //3.2.3
        QString patient_sex(data_header->patient_sex.c_str());
        x = width - _painter->fontMetrics().width(patient_sex) - border;
        _painter->drawText(x , (iItem++)*margin , patient_sex);

        //3.2.4
        std::stringstream ss;
        ss << data_header->columns << " " << data_header->rows << " " << data_header->slice_location.size();
        QString dim(ss.str().c_str());
        x = width - _painter->fontMetrics().width(dim) - border;
        _painter->drawText(x , (iItem++)*margin , dim);

        //////////////////////////////////////////////////////////////////////////
        //3.3 Left Bottom

        //Window level
        std::shared_ptr<MPRScene> mpr_scene = std::dynamic_pointer_cast<MPRScene>(_scene);
        if (mpr_scene)
        {
            float ww(1) , wl(0);
            mpr_scene->get_global_window_level(ww , wl);
            int wl_int = (int)wl;
            int ww_int = (int)ww;
            StrNumConverter<int> num_to_str;
            std::string wl_string = std::string("C : ") + num_to_str.to_string(wl_int) + std::string("  W : ") + num_to_str.to_string(ww_int); 
            iItem = 0;
            _painter->drawText(border , height - (iItem++)*margin - 2 , wl_string.c_str());
        }
        

    }
    catch (const Exception& e)
    {
        std::cout << e.what();
        //assert(false);
        throw e;
    }
}

MED_IMAGING_END_NAMESPACE