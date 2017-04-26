#ifndef MYWINDOW_H
#define MYWINDOW_H

#include <QMainWindow>
#include "ui_my_window.h"

namespace medical_imaging
{
    class MPRScene;
    class VolumeInfos;
}

class MyGraphicScene;
class MyGraphicView;


class MyWindow : public QMainWindow
{
    Q_OBJECT

public:
    MyWindow(QWidget *parent = 0);
    ~MyWindow();

protected:
    void configure();
    void create_scenes();

private slots:
    //File
    void slot_open_dicom_folder();

private:
    Ui::MainWindow ui;

private:
    MyGraphicView* _my_graphic_view[4];
    //MyGraphicScene* _my_graphic_scene[4];

    std::shared_ptr<medical_imaging::VolumeInfos> _volume_infos;

    std::shared_ptr<medical_imaging::MPRScene> _mpr_scenes[3];
};

#endif