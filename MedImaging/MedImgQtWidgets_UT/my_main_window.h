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
    virtual void closeEvent(QCloseEvent * pEvent);

protected slots: 
    void SlotAddDcmPostfix_i();

private:
    Ui::MedImgQtWidgets_UTClass ui;
    std::shared_ptr<TextEditerModule> m_pTextEditerModule;
    std::shared_ptr<MedicalImageDataModule> m_pMedImgModule;
    QGridLayout* m_pGridLayout;
    QHBoxLayout* m_pHBoxLayout;
    MyGLWidget* m_pMyGLWidget;
    QPlainTextEdit* m_pPlainTextEdit;

    //Scene
    SceneContainer* m_pMPRScene;

};

#endif // MEDIMGQTWIDGETS_UT_H
