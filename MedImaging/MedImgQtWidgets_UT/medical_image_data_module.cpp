#include "medical_image_data_module.h"

#include <iostream>

#include "MedImgCommon/mi_concurrency.h"
#include "MedImgCommon/mi_configuration.h"

#include "MedImgIO/mi_dicom_loader.h"
#include "MedImgIO/mi_image_data.h"
#include "MedImgIO/mi_image_data_header.h"
#include "MedImgIO/mi_meta_object_loader.h"

#include "MedImgArithmetic/mi_ortho_camera.h"

#include "MedImgGLResource/mi_gl_fbo.h"
#include "MedImgGLResource/mi_gl_texture_2d.h"

#include "MedImgRenderAlgorithm/mi_camera_calculator.h"
#include "MedImgRenderAlgorithm/mi_mpr_entry_exit_points.h"
#include "MedImgRenderAlgorithm/mi_ray_caster_canvas.h"
#include "MedImgRenderAlgorithm/mi_ray_caster.h"
#include "MedImgRenderAlgorithm/mi_camera_interactor.h"
#include "MedImgRenderAlgorithm/mi_volume_infos.h"
#include "MedImgRenderAlgorithm/mi_mpr_scene.h"

#include "MedImgQtWidgets/mi_scene_container.h"
#include "MedImgQtWidgets/mi_painter_corners_info.h"
#include "MedImgQtWidgets/mi_mouse_op_zoom.h"
#include "MedImgQtWidgets/mi_mouse_op_pan.h"
#include "MedImgQtWidgets/mi_mouse_op_rotate.h"
#include "MedImgQtWidgets/mi_mouse_op_mpr_paging.h"
#include "MedImgQtWidgets/mi_mouse_op_windowing.h"
#include "MedImgQtWidgets/mi_mouse_op_probe.h"

#include "qt/qaction.h"
#include "qt/qfiledialog.h"
#include "qt/qevent.h"

#include "my_main_window.h"

using namespace medical_imaging;

MedicalImageDataModule::MedicalImageDataModule(
    MyMainWindow*main_window , 
    SceneContainer* scene , 
    QAction* action_open_dicom , 
    QAction* action_open_meta,
    QObject* parent):QObject(parent)
{
    _main_window = main_window;
    m_pMPRScene = scene;
    connect(action_open_dicom , SIGNAL(triggered()) , this , SLOT(slot_action_open_dicom_folder()));
    connect(action_open_meta , SIGNAL(triggered()) , this , SLOT(slot_action_open_meta_folder()));
}

void MedicalImageDataModule::slot_action_open_dicom_folder()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        _main_window ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    std::vector<QString> vecFileNames = fileNames.toVector().toStdVector();
    if (!vecFileNames.empty())
    {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        std::vector<std::string> vecSTDFiles;
        for (auto it = vecFileNames.begin() ; it != vecFileNames.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            std::cout << s << std::endl;
            vecSTDFiles.push_back(s);
        }

        std::shared_ptr<ImageDataHeader> data_header;
        std::shared_ptr<ImageData> image_data;
        DICOMLoader loader;
        IOStatus status = loader.load_series(vecSTDFiles, image_data , data_header);

        m_pVolumeInfos.reset(new VolumeInfos());
        m_pVolumeInfos->set_data_header(data_header);
        m_pVolumeInfos->set_volume(image_data);

        std::shared_ptr<medical_imaging::MPRScene> scene(new medical_imaging::MPRScene(m_pMPRScene->width() , m_pMPRScene->height()));

        //m_pMPRScene->makeCurrent();
        scene->set_volume_infos(m_pVolumeInfos);
        scene->set_sample_rate(1.0);
        scene->set_global_window_level(252,40);
        scene->set_composite_mode(COMPOSITE_AVERAGE);
        scene->place_mpr(TRANSVERSE);
        m_pMPRScene->set_scene(scene);


        //Add painter list
        std::shared_ptr<CornersInfoPainter> pPatientInfo(new CornersInfoPainter());
        pPatientInfo->set_scene(scene);
        m_pMPRScene->add_painter_list(std::vector<std::shared_ptr<PainterBase>>(1 , pPatientInfo));


        //Add operation 
        std::shared_ptr<MouseOpZoom> pZoom(new MouseOpZoom());
        pZoom->set_scene(scene);
        m_pMPRScene->register_mouse_operation(pZoom , Qt::RightButton , Qt::NoModifier);

        std::shared_ptr<MouseOpRotate> pRotate(new MouseOpRotate());
        pRotate->set_scene(scene);
        m_pMPRScene->register_mouse_operation(pRotate , Qt::LeftButton , Qt::NoModifier);

        std::shared_ptr<MouseOpWindowing> pWindowing(new MouseOpWindowing());
        pWindowing->set_scene(scene);
        m_pMPRScene->register_mouse_operation(pWindowing , Qt::MiddleButton , Qt::NoModifier);

        std::shared_ptr<MouseOpPan> pPan(new MouseOpPan());
        pPan->set_scene(scene);
        m_pMPRScene->register_mouse_operation(pPan , Qt::MiddleButton , Qt::ControlModifier);

        std::shared_ptr<MouseOpProbe> pProbe(new MouseOpProbe());
        pProbe->set_scene(scene);
        m_pMPRScene->register_mouse_operation(pProbe , Qt::NoButton, Qt::NoModifier);


        QApplication::restoreOverrideCursor();


        _main_window->setWindowTitle(tr("MPR Scene"));
        m_pMPRScene->update();
    }
}

void MedicalImageDataModule::slot_action_open_meta_folder()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        _main_window ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    std::vector<QString> vecFileNames = fileNames.toVector().toStdVector();
    std::vector<std::shared_ptr<MetaObjectTag>> vecMetaObjTag;
    if (!vecFileNames.empty())
    {
        MetaObjectLoader loader;
        std::shared_ptr<ImageData> pImg;
        std::shared_ptr<ImageDataHeader> pImgHeader;

        std::vector<std::string> vecSTDFiles;
        for (auto it = vecFileNames.begin() ; it != vecFileNames.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            std::cout << s << std::endl;
            std::shared_ptr<MetaObjectTag> meta_obj_tag;
            loader.load(s , pImg ,meta_obj_tag , pImgHeader );
            vecMetaObjTag.push_back(meta_obj_tag);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //statistic
    double dCount = (double)vecMetaObjTag.size();

    double dSpacingXYMin = 65535.0;
    double dSpacingXYMax = -65535.0;
    double dSpacingXYMean = 0.0;
    
    double dSpacingZMin = 65535.0;
    double dSpacingZMax = -65535.0;
    double dSpacingZMean = 0.0;

    double dFovXYMin = 65535.0;
    double dFovXYMax = -65535.0;
    double dFovXYMean = 0.0;

    double dFovZMin = 65535.0;
    double dFovZMax = -65535.0;
    double dFovZMean = 0.0;

    for (size_t i = 0 ; i<vecMetaObjTag.size() ;++i)
    {
        double dSpacingXY = vecMetaObjTag[i]->spacing[0];
        double dSpacingZ = vecMetaObjTag[i]->spacing[2];
        double dFovXY = dSpacingXY*vecMetaObjTag[i]->dim_size[0];
        double dFovZ= dSpacingZ*vecMetaObjTag[i]->dim_size[2];

        dSpacingXYMin = dSpacingXYMin < dSpacingXY ? dSpacingXYMin : dSpacingXY;
        dSpacingXYMax = dSpacingXYMax > dSpacingXY ? dSpacingXYMax : dSpacingXY;

        dSpacingZMin = dSpacingZMin < dSpacingZ ? dSpacingZMin : dSpacingZ;
        dSpacingZMax = dSpacingZMax > dSpacingZ ? dSpacingZMax : dSpacingZ;

        dFovXYMin = dFovXYMin < dFovXY ? dFovXYMin : dFovXY;
        dFovXYMax = dFovXYMax > dFovXY ? dFovXYMax : dFovXY;

        dFovZMin = dFovZMin < dFovZ ? dFovZMin : dFovZ;
        dFovZMax = dFovZMax > dFovZ ? dFovZMax : dFovZ;

        dSpacingXYMean += dSpacingXY;
        dFovXYMean += dFovXY;

        dSpacingZMean += dSpacingZ;
        dFovZMean += dFovZ;
    }

    dSpacingXYMean/=dCount;
    dSpacingZMean/=dCount;
    dFovXYMean/=dCount;
    dFovZMean/=dCount;

    std::cout << "Max XY spacing : " << dSpacingXYMax << std::endl;
    std::cout << "Min XY spacing : " << dSpacingXYMin << std::endl;
    std::cout << "Average XY spacing : " << dSpacingXYMean << std::endl;
    std::cout << "Max Z spacing : " << dSpacingZMax << std::endl;
    std::cout << "Min Z spacing : " << dSpacingZMin << std::endl;
    std::cout << "Average Z spacing : " << dSpacingZMean << std::endl;
    std::cout << "Max XY Fov : " << dFovXYMax << std::endl;
    std::cout << "Min XY Fov : " << dFovXYMin << std::endl;
    std::cout << "Average XY Fov : " << dFovXYMean << std::endl;
    std::cout << "Max Z Fov : " << dFovZMax << std::endl;
    std::cout << "Min Z Fov : " << dFovZMin << std::endl;
    std::cout << "Average Z Fov : " << dFovZMean << std::endl;

    std::cout << "Done\n";
}
