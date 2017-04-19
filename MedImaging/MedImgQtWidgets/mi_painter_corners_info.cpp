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

void CornersInfoPainter::Render()//TODO patient four corners info by configuration way
{
    try
    {
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_pScene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(m_pPainter);

        std::shared_ptr<RayCastScene> pScene = std::dynamic_pointer_cast<RayCastScene>(m_pScene);
        QTWIDGETS_CHECK_NULL_EXCEPTION(pScene);

        std::shared_ptr<VolumeInfos> pVolumeInfos = pScene->GetVolumeInfos();
        QTWIDGETS_CHECK_NULL_EXCEPTION(pVolumeInfos);

        std::shared_ptr<ImageDataHeader> pDataHeader = pVolumeInfos->GetDataHeader();
        QTWIDGETS_CHECK_NULL_EXCEPTION(pDataHeader );

        int iWidth(1),iHeight(1);
        pScene->GetDisplaySize(iWidth , iHeight);

        //1 Set font
        const int iPointSize = 12;
        const int iMargin = iPointSize+2;
        const int iBorder = 3;
        QFont serifFont("Times" , iPointSize , QFont::Bold);
        m_pPainter->setFont(serifFont);

        //2 Set color
        m_pPainter->setPen(QColor(220,220,220));

        //////////////////////////////////////////////////////////////////////////
        //3 Patient four corners info
        //////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////
        //3.1 Left Top
        int iItem = 1;
        //3.1.1
        m_pPainter->drawText(iBorder , (iItem++)*iMargin , pDataHeader->m_sManufacturer.c_str());
        //3.1.2
        m_pPainter->drawText(iBorder , (iItem++)*iMargin , pDataHeader->m_sManufacturerModelName.c_str());
        //3.1.3
        std::string sModality = "UnSupported";
        switch(pDataHeader->m_eModality)
        {
        case CR:
            sModality = "CR";
            break;
        case CT:
            sModality = "CT";
            break;
        case MR:
            sModality = "MR";
            break;
        case PT:
            sModality = "PT";
            break;
        default:
            break;
        }
        m_pPainter->drawText(iBorder , (iItem++)*iMargin , sModality.c_str());
        //3.1.4
        m_pPainter->drawText(iBorder , (iItem++)*iMargin , pDataHeader->m_sImageDate.c_str());

        //////////////////////////////////////////////////////////////////////////
        //3.2 Right Top
        iItem = 1;
        int iX = 0;
        //3.2.1
        QString sPatientName(pDataHeader->m_sPatientName.c_str());
        iX = iWidth - m_pPainter->fontMetrics().width(sPatientName) - iBorder;
        m_pPainter->drawText(iX , (iItem++)*iMargin , sPatientName);

        //3.2.2
        QString sPatientID(pDataHeader->m_sPatientID.c_str());
        iX = iWidth - m_pPainter->fontMetrics().width(sPatientID) - iBorder;
        m_pPainter->drawText(iX , (iItem++)*iMargin , sPatientID);

        //3.2.3
        QString sPatientSex(pDataHeader->m_sPatientSex.c_str());
        iX = iWidth - m_pPainter->fontMetrics().width(sPatientSex) - iBorder;
        m_pPainter->drawText(iX , (iItem++)*iMargin , sPatientSex);

        //3.2.4
        std::stringstream ss;
        ss << pDataHeader->m_uiImgColumns << " " << pDataHeader->m_uiImgRows << " " << pDataHeader->m_SliceLocations.size();
        QString sDim(ss.str().c_str());
        iX = iWidth - m_pPainter->fontMetrics().width(sDim) - iBorder;
        m_pPainter->drawText(iX , (iItem++)*iMargin , sDim);

        //////////////////////////////////////////////////////////////////////////
        //3.3 Left Bottom

        //Window level
        std::shared_ptr<MPRScene> pMPRScene = std::dynamic_pointer_cast<MPRScene>(m_pScene);
        if (pMPRScene)
        {
            float fWW(1) , fWL(0);
            pMPRScene->GetGlobalWindowLevel(fWW , fWL);
            int iWL = (int)fWL;
            int iWW = (int)fWW;
            StrNumConverter<int> numToStr;
            std::string sWL = std::string("C : ") + numToStr.ToString(iWL) + std::string("  W : ") + numToStr.ToString(iWW); 
            iItem = 0;
            m_pPainter->drawText(iBorder , iHeight - (iItem++)*iMargin - 2 , sWL.c_str());
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