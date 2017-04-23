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

    _my_gl_widget = new MyGLWidget();
    _my_gl_widget->setMinimumSize(100,100);
    _my_gl_widget->resize(200,200);
    _my_gl_widget->setFixedSize(300,300);

    _plain_text_edit = new QPlainTextEdit();
    _plain_text_edit->setMinimumSize(100,100);
    //m_pPlainTextEdit->setFixedSize(200,200);

    _grid_layout= new QGridLayout();
    _hbox_layout = new QHBoxLayout();
    /*m_pGridLayout->setSpacing(2);
    m_pGridLayout->setMargin(2);
    m_pGridLayout->addLayout(m_pHBoxLayout);*/

    QPushButton* pTest = new QPushButton();

    _mpr_container = new SceneContainer(SharedWidget::instance());
    _mpr_container->setMinimumSize(500,500);
    //m_pMPRScene->setFixedSize(300,300);

    //m_pGridLayout->addWidget(m_pPlainTextEdit , 0 ,0);
    //m_pGridLayout->addWidget(m_pMyGLWidget,0,1);
    //m_pGridLayout->addWidget(pTest , 1,0);
    //m_pGridLayout->addWidget(m_pMPRScene , 1 , 1);
    _grid_layout->addWidget(_mpr_container , 0 , 0);

    ui.centralWidget->setLayout(_grid_layout);

    _text_editer_module.reset(new TextEditerModule(this , 
        ui.actionNew,
        ui.actionOpen,
        ui.actionSave,
        ui.actionSaveAs,
        ui.actionUndo,
        ui.actionRedo,
        ui.actionCut,
        ui.actionCopy,
        ui.actionPaste,
        _plain_text_edit));

    _med_img_module.reset( new MedicalImageDataModule(this ,
        _mpr_container , 
        ui.actionOpen_DICOM_floder,
        ui.actionOpen_Meta_Folder));

    connect(ui.actionAdd_dcm_postfix , SIGNAL(triggered()) , this , SLOT(SlotAddDcmPostfix_i()));
}

MyMainWindow::~MyMainWindow()
{

}

void MyMainWindow::closeEvent( QCloseEvent * event )
{
    if (_text_editer_module->close_window())
    {
        event->accept();
    }
    else
    {
        event->ignore();
    }
}

void MyMainWindow::SlotAddDcmPostfix_i()
{
    QStringList file_name_list = QFileDialog::getOpenFileNames(
        this ,tr("Loading DICOM Dialog"),"",tr("Dicom image(*dcm);;Other(*)"));

    std::vector<QString> file_names = file_name_list.toVector().toStdVector();
    if (!file_names.empty())
    {

        std::vector<std::string> file_names_std;
        for (auto it = file_names.begin() ; it != file_names.end() ; ++it)
        {
            std::string s((*it).toLocal8Bit());
            std::cout << s << std::endl;
            file_names_std.push_back(s);
            rename(s.c_str() , (s+".dcm").c_str());
        }
    }
}
