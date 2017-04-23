#ifndef MEDIMGQTWIDGETS_UT_H
#define MEDIMGQTWIDGETS_UT_H

#include "gl/glew.h"
#include <QtGui/QMainWindow>
#include "ui_medimgqtwidgets_ut.h"

class QCloseEvent;
class QPlainTextEdit;
class QGridLayout;
class QHBoxLayout;

class TextEditerModule;
class MedicalImageDataModule;
class MyGLWidget;
class MPRScene;
class SceneContainer;


class MyMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MyMainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~MyMainWindow();

protected:
    virtual void closeEvent(QCloseEvent * event);

protected slots: 
    void SlotAddDcmPostfix_i();

private:
    Ui::MedImgQtWidgets_UTClass ui;
    std::shared_ptr<TextEditerModule> _text_editer_module;
    std::shared_ptr<MedicalImageDataModule> _med_img_module;
    QGridLayout* _grid_layout;
    QHBoxLayout* _hbox_layout;
    MyGLWidget* _my_gl_widget;
    QPlainTextEdit* _plain_text_edit;

    //Scene
    SceneContainer* _mpr_container;

};

#endif // MEDIMGQTWIDGETS_UT_H
