#include "my_main_window.h"
#include <iostream>
#include "qt/qfiledialog.h"
#include "qt/qevent.h"

#include "qt/qplaintextedit.h"
#include "qt/qgridlayout.h"
#include "qt/qboxlayout.h"
#include "qt/qpushbutton.h"

#include "text_editer_module.h"
#include "my_gl_widget.h"
#include "medical_image_data_module.h"

#include "MedImgCommon/mi_configuration.h"
#include "MedImgGLResource/mi_gl_utils.h"
#include "MedImgQtWidgets/mi_shared_widget.h"
#include "MedImgQtWidgets/mi_scene_container.h"

using namespace medical_imaging;

MyMainWindow::MyMainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
    //Set configuration
    Configuration::instance()->set_processing_unit_type(GPU);
    GLUtils::set_check_gl_flag(false);

    ui.setupUi(this);

    this->resize(1000,800);
    this->setMinimumSize(500,500);
    //Set background color
    /*QPalette pal(ui.centralWidget->palette());
    pal.setColor(QPalette::Background ,  Qt::black);
    ui.centralWidget->setPalette(pal);
    ui.centralWidget->setAutoFillBackground(true);*/

    //this->setCentralWidget(ui.plainTextEdit);

    m_pMyGLWidget = new MyGLWidget();
    m_pMyGLWidget->setMinimumSize(100,100);
    m_pMyGLWidget->resize(200,200);
    m_pMyGLWidget->setFixedSize(300,300);

    m_pPlainTextEdit = new QPlainTextEdit();
    m_pPlainTextEdit->setMinimumSize(100,100);
    //m_pPlainTextEdit->setFixedSize(200,200);

    m_pGridLayout= new QGridLayout();
    m_pHBoxLayout = new QHBoxLayout();
    /*m_pGridLayout->setSpacing(2);
    m_pGridLayout->setMargin(2);
    m_pGridLayout->addLayout(m_pHBoxLayout);*/

    QPushButton* pTest = new QPushButton();

    m_pMPRScene = new SceneContainer(SharedWidget::instance());
    m_pMPRScene->setMinimumSize(500,500);
    //m_pMPRScene->setFixedSize(300,300);

    //m_pGridLayout->addWidget(m_pPlainTextEdit , 0 ,0);
    //m_pGridLayout->addWidget(m_pMyGLWidget,0,1);
    //m_pGridLayout->addWidget(pTest , 1,0);
    //m_pGridLayout->addWidget(m_pMPRScene , 1 , 1);
    m_pGridLayout->addWidget(m_pMPRScene , 0 , 0);

    ui.centralWidget->setLayout(m_pGridLayout);

    m_pTextEditerModule.reset(new TextEditerModule(this , 
        ui.actionNew,
        ui.actionOpen,
        ui.actionSave,
        ui.actionSaveAs,
        ui.actionUndo,
        ui.actionRedo,
        ui.actionCut,
        ui.actionCopy,
        ui.actionPaste,
        m_pPlainTextEdit));

    m_pMedImgModule.reset( new MedicalImageDataModule(this ,
        m_pMPRScene , 
        ui.actionOpen_DICOM_floder,
        ui.actionOpen_Meta_Folder));

    connect(ui.actionAdd_dcm_postfix , SIGNAL(triggered()) , this , SLOT(SlotAddDcmPostfix_i()));
}

MyMainWindow::~MyMainWindow()
{

}

void MyMainWindow::closeEvent( QCloseEvent * pEvent )
{
    if (m_pTextEditerModule->close_window())
    {
        pEvent->accept();
    }
    else
    {
        pEvent->ignore();
    }
}

void MyMainWindow::SlotAddDcmPostfix_i()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
        this ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    std::vector<QString> vecFileNames = fileNames.toVector().toStdVector();
    if (!vecFileNames.empty())
    {

        std::vector<std::string> vecSTDFiles;
        for (auto it = vecFileNames.begin() ; it != vecFileNames.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            std::cout << s << std::endl;
            vecSTDFiles.push_back(s);
            rename(s.c_str() , (s+".dcm").c_str());
        }
    }
}
