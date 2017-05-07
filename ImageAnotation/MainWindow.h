/**
* The Image Annotation Tool for image annotations with pixelwise masks
*
* Copyright (C) 2007 Alexander Klaeser
*
* http://lear.inrialpes.fr/people/klaeser/
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#ifndef MainWindow_H
#define MainWindow_H

#include <QMainWindow>
#include <QRectF>
#include <QCloseEvent>
#include <QStringList>
#include <QVector>
#include <QList>
#include <QColor>
#include "ui_MainWindow.h"
#include "PixmapWidget.h"
#include "ImgAnnotation.h"
#include "ScrollAreaNoWheel.h"


class MainWindow : public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT

public:
    enum Direction { Up, Down };

public:
    MainWindow(QWidget *parent = 0, QFlag flags = 0);
    QString getMaskFile(int iMask, QString fileName) const;
    QString currentDir() const;
    QString currentFile() const;
    QString currentObjFile() const;
    int currentObj() const;

protected:
    void closeEvent(QCloseEvent *event);
    void keyPressEvent(QKeyEvent * event);
    void keyReleaseEvent(QKeyEvent * event);
    void wheelEvent(QWheelEvent *event);

private:
    void errorMessageMask();
    void saveMask();
    void updateUndoMenu();

private slots:
    void onBoundingBoxTransformed(BoundingBox);
    void onActiveBoundingBoxChanged(int);
    void onWheelTurnedInScrollArea(QWheelEvent *);
    void on_actionOpenDir_triggered();
    void on_actionQuit_triggered();
    void on_actionShortcutHelp_triggered();
    void on_actionUndo_triggered();
    void on_actionRedo_triggered();
    void on_addObjButton_clicked();
    void on_delObjButton_clicked();
    void on_duplicateObjButton_clicked();
    void on_transparencySlider_valueChanged(int i);
    void on_objTypeComboBox_currentIndexChanged(const QString &);
    void on_imgTreeWidget_currentItemChanged(QTreeWidgetItem *, QTreeWidgetItem *);
    void on_objListWidget_currentItemChanged(QListWidgetItem *, QListWidgetItem *);
    void on_brushSizeComboBox_currentIndexChanged(int);
    void on_tagListWidget_itemSelectionChanged();
    void on_drawOnTypeComboBox_currentIndexChanged(int);
    void onMaskDraw(QImage *mask);
    void refreshImgView();
    void refreshObjView();
    void refreshTagView();
    void refreshObjMask();
    void nextPreviousFile(Direction);

private:
    PixmapWidget *pixmapWidget;
    ScrollAreaNoWheel *scrollArea;
    QString currentlyOpenedDir;
    QStringList currentObjFilenames;

    ImgAnnotation annotations;
    QString lastObjType;

    IAObj copiedObj;

    QVector<QRgb> colorTable;
    int iBackgroundColor, iObjectColor, iOccludedColor;
    QStringList objTypes;
    QStringList maskTypes;
    QStringList labels;
    int iBackgroundMask, iObjMask, iOccludedObjMask;
    QVector<int> brushSizes;

    QList<QImage> imgUndoHistory;
    int currentHistoryImg;
    int maxHistorySize;

    bool keyShiftPressed;
    bool keyCtrlPressed;
};

#endif
