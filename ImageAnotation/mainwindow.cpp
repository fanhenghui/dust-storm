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
#include "MainWindow.h"
#include <iostream>
#include <string>

#include <QFileDialog>
#include <QDir>
#include <QFile>
#include <QStringList>
#include <QMessageBox>
#include <stdio.h>
#include "ScrollAreaNoWheel.h"
#include <QtDebug>
#include <QMessageBox>
#include <QImageReader>


#define MASK_TPYE_NUM 9
static const std::string S_mask_types[MASK_TPYE_NUM] = 
{
    "none",
    "microaneurysms" , 
    "exudates",
    "hemorrhages",
    "cotton wool spots",
    "venous beading",
    "neovascularization",
    "IMRA"
    "hemorrhages spot"
};


bool maskFileLessThan(const QString &f1, const QString &f2)
{
    // mask file name looks like: <imageFileName>.mask.<Number>.<extension>
    // compare front part
    QString front1 = f1.section(".", 0, -4);
    QString front2 = f2.section(".", 0, -4);
    if (front1 != front2)
        return front1 < front2;

    // compare numbers
    QString strNum1 = f1.section(".", -2, -2);
    QString strNum2 = f2.section(".", -2, -2);
    bool ok1;
    int num1 = strNum1.toInt(&ok1);
    bool ok2;
    int num2 = strNum2.toInt(&ok2);

    if (!ok1 || num1 < 0)
        return false;
    if (!ok2 || num2 < 0)
        return true;
    return num1 < num2;
}


MainWindow::MainWindow(QWidget *parent, QFlag flags)
    : QMainWindow(parent, flags)
{
    // set up the UI
    setupUi(this);
    scrollArea = new ScrollAreaNoWheel(this);
    pixmapWidget = new PixmapWidget(scrollArea, scrollArea);
    pixmapWidget->setFloodFill(true);
    
    scrollArea->setWidgetResizable(true);
    scrollArea->setWidget(pixmapWidget);
    setCentralWidget(scrollArea);
    keyShiftPressed = false;
    keyCtrlPressed = false;

    // some hardcoded data
    colorTable << qRgb(0, 0, 0); // background
    colorTable << qRgb(255, 0, 0); // object
    colorTable << qRgb(0, 255, 0); // occluded
    iBackgroundColor = 0;
    iObjectColor = 1;
    iOccludedColor = 2;
    objTypes << "Default";
    maskTypes << "background" << "object" << "occluded object";
    labels << "positive" << "uncertain";
    iBackgroundMask = 0;
    iObjMask = 1;
    iOccludedObjMask = 2;
    brushSizes << 1 << 3 << 5 << 7 << 9 << 11 << 13 << 15 << 18 << 20 << 25 << 30 << 50 << 100;
    maxHistorySize = 10;

    currentHistoryImg = 0;

    for (int i = 0; i < brushSizes.size(); i++)
        brushSizeComboBox->addItem("Circle (" + QString::number(brushSizes[i]) + "x" + QString::number(brushSizes[i]) + ")");


    // we want to receive key events, therefore we have to set the focus policy
    setFocusPolicy(Qt::WheelFocus);

    // make some connections
    connect(pixmapWidget, SIGNAL(drawEvent(QImage *)), this, SLOT(onMaskDraw(QImage *)));
    connect(zoomSpinBox, SIGNAL(valueChanged(double)), pixmapWidget, SLOT(setZoomFactor(double)));
    connect(pixmapWidget, SIGNAL(zoomFactorChanged(double)), zoomSpinBox, SLOT(setValue(double)));
    connect(scrollArea, SIGNAL(wheelTurned(QWheelEvent*)), this, SLOT(onWheelTurnedInScrollArea(QWheelEvent *)));

    // set some default values
    brushSizeComboBox->setCurrentIndex(1);
}

QString MainWindow::getMaskFile(int iMask, QString fileName) const
{
    return fileName.replace(".image.", ".").section(".", 0, -2) + ".mask." + QString(S_mask_types[iMask].c_str()) + ".png";
}

QString MainWindow::currentDir() const
{
    QTreeWidgetItem *current = imgTreeWidget->currentItem();
    if (!current || !current->parent())
        return "";

    QString dir = current->parent()->text(0);

    return dir;
}

QString MainWindow::currentFile() const
{
    QTreeWidgetItem *current = imgTreeWidget->currentItem();
    if (!current || !current->parent())
        return "";

    return current->text(0);
}

QString MainWindow::currentObjFile()
{
    const int iObj = currentObj();
    if (iObj < 1)
    {
        return QString("");
    }
    else
    {
        return _current_obj_file_collection[iObj];
    }
}

int MainWindow::currentObj() const
{
    QString file = currentFile();
    QString dir = currentDir();
    if (file.isEmpty() || dir.isEmpty())
    {
        return -1;
    }

    return ComboBox_ObjType->currentIndex();
}


void MainWindow::onWheelTurnedInScrollArea(QWheelEvent *event)
{
    wheelEvent(event);
}

void MainWindow::on_actionOpenDir_triggered()
{
    // clear the status bar and set the normal mode for the pixmapWidget
    statusBar()->clearMessage();

    // ask the user to add files
    QString openedDir = QFileDialog::getExistingDirectory(this, "Choose a directory to be read in", currentlyOpenedDir);

    if (openedDir.isEmpty())
        return;

    // save the opened path
    currentlyOpenedDir = openedDir;

    // read in the directory structure
    refreshImgView();

    // update the window title
    setWindowTitle("ImageAnnotation - " + openedDir);

    // the ctrl key rests
    keyCtrlPressed = false;
    keyShiftPressed = false;
    pixmapWidget->setFloodFill(false);

    // update the statusbar
    statusBar()->showMessage("Opened directory structure " + openedDir, 5 * 1000);
}

void MainWindow::on_actionQuit_triggered()
{
    close();
}

void MainWindow::on_actionShortcutHelp_triggered()
{
    // clear the status bar and set the normal mode for the pixmapWidget
    statusBar()->clearMessage();

    // we display an overview on shortcuts
    QMessageBox::about(this, "Shortcut Help",
        "<table border=0 cellpadding=0 cellspacing=2>\n"
        "<tr>\n"
        "<td><b>Left Mouse Button</b></td>\n"
        "<td width=10></td>\n"
        "<td>draw in the chosen color</td>\n"
        "</tr><tr>\n"
        "<td><b>Right Mouse Button</b></td>\n"
        "<td width=10></td>\n"
        "<td>draw a line from the last drawn position to the current one</td>\n"
        "</tr><tr>\n"
        "<td><b>Ctrl + Left Mouse Button</b></td>\n"
        "<td width=10></td>\n"
        "<td>flood filling with the current color</td>\n"
        "</tr><tr>\n"
        "<td><b>Alt+A</b></td>\n"
        "<td width=10></td>\n"
        "<td>add a new object</td>\n"
        "</tr><tr>\n"
        "<td><b>Alt+D</b></td>\n"
        "<td width=10></td>\n"
        "<td>duplicate the current object</td>\n"
        "</tr><tr>\n"
        "<td><b>Alt+R</b></td>\n"
        "<td width=10></td>\n"
        "<td>remove the current object</td>\n"
        "</tr><tr>\n"
        "<td><b>1, ..., 9</b></td>\n"
        "<td></td>\n"
        "<td>choose brush size from drop down box</td>\n"
        "</tr><tr>\n"
        "<td><b>F1, F2, F3</b></td>\n"
        "<td></td>\n"
        "<td>choose edit color from drop down box</td>\n"
        "</tr><tr>\n"
        "<td><b>Shift+F1, ... F4</b></td>\n"
        "<td></td>\n"
        "<td>choose draw on color from drop down box</td>\n"
        "</tr><tr>\n"
        "<td><b>MouseWheel Up/Down</b></td>\n"
        "<td></td>\n"
        "<td>zoom out/in</td>\n"
        "</tr><tr>\n"
        "<td><b>Ctrl+MouseWheel Up/Down</b></td>\n"
        "<td></td>\n"
        "<td>increase/decrease brush size</td>\n"
        "</tr><tr>\n"
        "<td><b>Shift+MouseWheel Up/Down</b></td>\n"
        "<td></td>\n"
        "<td>go to the previous/next file in the file list</td>\n"
        "</tr>\n"
        "</table>\n");
}

void MainWindow::on_actionUndo_triggered()
{
    if (currentHistoryImg < imgUndoHistory.size() - 1 && imgUndoHistory.size() > 1) {
        // get the name of the mask image file
        QString iFile = currentFile();
        QString iDir = currentDir();
        QString iMask = currentObjFile();
        if (iFile.isEmpty() || iDir.isEmpty() || iMask.isEmpty())
            return;
        QString objMaskFilename = getMaskFile(currentObj(), iFile);

        // save the image from the history
        currentHistoryImg++;
        if (!imgUndoHistory[currentHistoryImg].save(currentlyOpenedDir + iDir + "/" + objMaskFilename, "PNG")) {
            errorMessageMask();
            return;
        }

        refreshObjMask();
        updateUndoMenu();
    }
}

void MainWindow::on_actionRedo_triggered()
{
    if (currentHistoryImg > 0 && currentHistoryImg < imgUndoHistory.size() && imgUndoHistory.size() > 1) {
        // get the name of the mask image file
        QString iFile = currentFile();
        QString iDir = currentDir();
        QString iMask = currentObjFile();
        if (iFile.isEmpty() || iDir.isEmpty() || iMask.isEmpty())
            return;
        QString objMaskFilename = getMaskFile(currentObj(), iFile);

        // save the image from the history
        currentHistoryImg--;
        if (!imgUndoHistory[currentHistoryImg].save(currentlyOpenedDir + iDir + "/" + objMaskFilename, "PNG")) {
            errorMessageMask();
            return;
        }

        refreshObjMask();
        updateUndoMenu();
    }
}

void MainWindow::on_ComboBox_ObjType_currentIndexChanged(int obj_id)
{

    if (0 == obj_id)
    {
        pixmapWidget->setFloodFill(true);
    }
    else
    {
        pixmapWidget->setFloodFill(false);

        QString iFile = currentFile();
        QString iDir = currentDir();
        if (iFile.isEmpty() || iDir.isEmpty() || obj_id == 0)
        {
            return;
        }

        //Check empty mask file
        const bool empty_obj_file = _current_obj_file_collection.find(obj_id) == _current_obj_file_collection.end();
        // create a new segmentation mask
        if (empty_obj_file)
        {
            QImage orgImg(currentlyOpenedDir + iDir + "/" + iFile);
            QImage mask(orgImg.size(), QImage::Format_Indexed8);
            mask.setColorTable(colorTable);
            mask.fill(iBackgroundColor);
            //mask.setText("annotationObjType", objTypes[0]);
            QString objMaskFilename = getMaskFile(obj_id, iFile);
            if (!mask.save(currentlyOpenedDir + iDir + "/" + objMaskFilename, "PNG")) 
            {
                errorMessageMask();
                return;
            }
            _current_obj_file_collection[obj_id] = objMaskFilename;
        }


        // refresh
        refreshObjMask();

        // clear the history and append the current mask
        imgUndoHistory.clear();
        QImage maskImg = pixmapWidget->getMask()->copy();
        imgUndoHistory.push_front(maskImg);
        currentHistoryImg = 0;
        updateUndoMenu();

    }
}

void MainWindow::on_imgTreeWidget_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    // check weather dir/file/object have been selected
    QString iFile = currentFile();
    QString iDir = currentDir();
    if (iFile.isEmpty() || iDir.isEmpty())
        return;

    // check weather we have a relative or absolute path
    QString absoluteDir;
    if (iDir[0] != '/')
        absoluteDir = currentlyOpenedDir;

    // load new file
    QString filepath(absoluteDir + iDir + "/" + iFile);
    pixmapWidget->setPixmap(QPixmap(filepath  ));

    // refresh the objListWidget
    getMaskFiles();
    if (_current_obj_file_collection.empty())
    {
        ComboBox_ObjType->blockSignals(true);
        ComboBox_ObjType->setCurrentIndex(0);
        ComboBox_ObjType->blockSignals(false);
    }
    else
    {
        ComboBox_ObjType->blockSignals(true);
        const int current_obj_id = _current_obj_file_collection.begin()->first;
        ComboBox_ObjType->setCurrentIndex(current_obj_id);
        ComboBox_ObjType->blockSignals(false);
    }

    currentHistoryImg = 0;

    refreshObjMask();
}


void MainWindow::on_brushSizeComboBox_currentIndexChanged(int i)
{
    if (i < 0 || i >= brushSizes.size())
        return;
    pixmapWidget->setPenWidth(brushSizes[i]);
}

void MainWindow::on_transparencySlider_valueChanged(int i)
{
    pixmapWidget->setMaskTransparency(((double)i) / transparencySlider->maximum());
}

void MainWindow::onMaskDraw(QImage *mask)
{
    // check weather dir/file/object have been selected
    QString iFile = currentFile();
    QString iDir = currentDir();
    int iObj = currentObj();
    if (iFile.isEmpty() || iDir.isEmpty() || iObj < 0)
        return;

    // save the mask
    // should only save mask once when finish mask draw
    saveMask();
}

void MainWindow::refreshImgView()
{
    QList<QByteArray> sup_imgs = QImageReader::supportedImageFormats();
    std::cout << "Support format : ";
    std::list<QByteArray> std_sup_imgs = sup_imgs.toStdList();
    for (auto it = std_sup_imgs.begin() ; it != std_sup_imgs.end() ; ++it)
    {
        std::cout << (*it).data() << " , ";
    }
    std::cout << std::endl;

    // clear all items
    imgTreeWidget->clear();

    // read in the currently opened directory structure recursively
    QStringList dirs;
    QStringList relativeDirStack;
    relativeDirStack << "."; // start with the main dir
    while (!relativeDirStack.empty()) {
        QString nextDirStr = relativeDirStack.first();
        relativeDirStack.pop_front();
        dirs << nextDirStr;

        // get all directories in the current directory
        QDir currentDir(currentlyOpenedDir + nextDirStr);
        QStringList dirList = currentDir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot);
        for (int i = 0; i < dirList.size(); i++) {
            relativeDirStack << nextDirStr + "/" + dirList[i];
        }
    }

    // read in all images in all collected directories
    // add all files and directories to the QTreeWidget
    QStringList nameFilters;
    nameFilters << "*.jpg" << "*.png" << "*.bmp" << "*.jpeg" << "*.tif" << "*.gif" << "*.tiff" << "*.pbm" << "*.pgm" << "*.ppm" << "*.xbm" << "*.xpm";
    for (int i = 0; i < dirs.size(); i++) {
        // get all images in the current directory
        QDir currentDir(currentlyOpenedDir + dirs[i]);
        currentDir.setFilter(QDir::Files);
        currentDir.setNameFilters(nameFilters);
        QStringList files = currentDir.entryList();

        // ignore the directory if we couldn't find any image file in it
        if (files.size() <= 0)
            continue;

        // construct a new directory entry
        QTreeWidgetItem *currentTreeDir = new QTreeWidgetItem(imgTreeWidget);
        imgTreeWidget->setItemExpanded(currentTreeDir, true);
        currentTreeDir->setText(0, dirs[i]);

        // construct new entries for the image files
        for (int j = 0; j < files.size(); j++) {
            // make sure that the image file is not a mask
            if (files[j].contains(".mask."))
                continue;

            // construct a new entry
            QTreeWidgetItem *currentFile = new QTreeWidgetItem(currentTreeDir);
            currentFile->setText(0, files[j]);
        }
    }

    // sort the files + directories
    imgTreeWidget->sortItems(0, Qt::AscendingOrder);
}

void MainWindow::refreshObjMask()
{
    // check wether dir/file/object have been selected
    QString iFile = currentFile();
    QString iDir = currentDir();
    int iObj = currentObj();
    if (iFile.isEmpty() || iDir.isEmpty() || iObj == 0)
        return;

    //if (iObj < 0) {
    //    // set an empty mask if no mask exists
    //    QImage tmp_org_img(currentlyOpenedDir + iDir + "/" + iFile);
    //    QImage emptyMask;
    //    pixmapWidget->setMask(emptyMask);
    //}
    //else 
    {
        // load the mask
        QImage mask(currentlyOpenedDir + iDir + "/" + currentObjFile());

        // convert binary masks
        if (mask.colorCount() == 2) {
            QImage newMask(mask.size(), QImage::Format_Indexed8);
            newMask.setColorTable(colorTable);
            for (int y = 0; y < newMask.height(); y++)
                for (int x = 0; x < newMask.width(); x++)
                    newMask.setPixel(x, y, mask.pixelIndex(x, y));
            mask = newMask;
        }

        pixmapWidget->setMask(mask);
    }
}

void MainWindow::nextPreviousFile(MainWindow::Direction direction)
{
    // choose the current items from the imgTreeWidget
    QTreeWidgetItem *current = imgTreeWidget->currentItem();
    if (!current)
        return;
    QTreeWidgetItem *currentParent = current->parent();

    if (!currentParent) {
        // we have a directory selected .. take the first file as current item
        current = current->child(0);
        currentParent = current->parent();
    }

    if (!current || !currentParent)
        return;

    // get the indeces
    int iParent = imgTreeWidget->indexOfTopLevelItem(currentParent);
    int iCurrent = currentParent->indexOfChild(current);

    // select the next file index
    if (direction == Up)
        iCurrent--;
    else
        iCurrent++;

    // the index may be negative .. in that case we switch the parent as well
    if (iCurrent < 0) {
        if (iParent > 0) {
            // get the directory before
            iParent--;
            currentParent = imgTreeWidget->topLevelItem(iParent);

            if (!currentParent)
                return;

            // get the last item from the directory before
            iCurrent = currentParent->childCount() - 1;
        }
        else
            // we are at the beginning ..
            iCurrent = 0;
    }
    // the index might be too large .. in that case we switch the parent as well
    else if (iCurrent >= currentParent->childCount()) {
        if (iParent < imgTreeWidget->topLevelItemCount() - 1) {
            // get the next directory
            iParent++;
            currentParent = imgTreeWidget->topLevelItem(iParent);

            if (!currentParent)
                return;

            // get the first item from the next directory
            iCurrent = 0;
        }
        else
            // we are at the end ..
            iCurrent = currentParent->childCount() - 1;
    }

    if (!currentParent)
        return;

    // we handled all special cases thus we may try to set the next current item
    current = currentParent->child(iCurrent);
    if (current)
        imgTreeWidget->setCurrentItem(current);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    event->accept();
}

void MainWindow::keyPressEvent(QKeyEvent * event)
{
    if (event->key() == Qt::Key_Control) {
        keyCtrlPressed = true;
        //pixmapWidget->setFloodFill(true);
        statusBar()->showMessage("Use mouse wheel to increase/decrease brush size");
        event->accept();
    }
    else if (event->key() == Qt::Key_Shift) {
        keyShiftPressed = true;
        statusBar()->showMessage("Use the mouse wheel to change files");
        event->accept();
    }
    // keys to change the color
    else if (keyShiftPressed && event->key() >= Qt::Key_F1 && event->key() <= Qt::Key_F4) {
        event->accept();
    }
    // keys to change the brush size
    else if (event->key() >= Qt::Key_1 && event->key() <= Qt::Key_9) {
        int index = event->key() - Qt::Key_1;
        if (index >= 0 && index < brushSizeComboBox->count())
            brushSizeComboBox->setCurrentIndex(index);
        event->accept();
    }
    else
        event->ignore();
}

void MainWindow::keyReleaseEvent(QKeyEvent * event)
{
    if (event->key() == Qt::Key_Control) {
        event->accept();
        keyCtrlPressed = false;
        //pixmapWidget->setFloodFill(false);
        statusBar()->clearMessage();
    }
    else if (event->key() == Qt::Key_Shift) {
        event->accept();
        keyShiftPressed = false;
        statusBar()->clearMessage();
    }
    else
        event->ignore();
}

void MainWindow::wheelEvent(QWheelEvent *event)
{
    if (!event->isAccepted()) {
        // see what to do with the event
        if (keyShiftPressed) {
            // select a different file
            if (event->delta() < 0)
                nextPreviousFile(Down);
            else if (event->delta() > 0)
                nextPreviousFile(Up);
        }
        else if (keyCtrlPressed) {
            // select a different object
            if (event->delta() > 0) {
                int idx = brushSizeComboBox->currentIndex() + 1;
                idx = idx > brushSizeComboBox->count() ? brushSizeComboBox->count() : idx;
                brushSizeComboBox->setCurrentIndex(idx);
            }
            else if (event->delta() < 0) {
                int idx = brushSizeComboBox->currentIndex() - 1;
                idx = idx > 0 ? idx : 0;
                brushSizeComboBox->setCurrentIndex(idx);
            }
        }
        else {
            // forward the wheelEvent to the zoomSpinBox
            if (event->delta() > 0)
                zoomSpinBox->stepDown();
            else if (event->delta() < 0)
                zoomSpinBox->stepUp();
        }
        event->accept();
    }
}

void MainWindow::errorMessageMask()
{
    QMessageBox::critical(this, "Writing Error", "Object mask files could not be changed/created.\nPlease check your user rights for directory and files.");
}

void MainWindow::saveMask()
{
    // check whether dir/file/object have been selected
    QString iFile = currentFile();
    QString iDir = currentDir();
    int iObj = currentObj();
    if (iFile.isEmpty() || iDir.isEmpty() || iObj ==0)
        return;

    // get the current mask
    QImage *mask = pixmapWidget->getMask();

    // save the mask
    mask->save(currentlyOpenedDir + iDir + "/" + _current_obj_file_collection[iObj], "PNG");

    // save the image in the history and delete items in case the history
    // is too big
    while (0 < currentHistoryImg) 
    {
        imgUndoHistory.pop_front();
        currentHistoryImg--;
    }
    QImage maskCopy = mask->copy();
    imgUndoHistory.push_front(maskCopy);
    while (imgUndoHistory.size() > maxHistorySize)
        imgUndoHistory.pop_back();

    updateUndoMenu();
}

void MainWindow::updateUndoMenu()
{
    // enable/disable the undo/redo menu items
    if (currentHistoryImg < imgUndoHistory.size()  && imgUndoHistory.size() > 1)
        actionUndo->setEnabled(true);
    else
        actionUndo->setEnabled(false);
    if (currentHistoryImg > 0 && imgUndoHistory.size() > 1)
        actionRedo->setEnabled(true);
    else
        actionRedo->setEnabled(false);
}

std::map<int, QString> MainWindow::getMaskFiles()
{
    // check weather dir/file/object have been selected
    QString iFile = currentFile();
    QString iDir = currentDir();
    if (iFile.isEmpty() || iDir.isEmpty())
        return std::map<int ,QString>();


    // find all mask images for the current image .. they determine the
    // number of objects for one image
    QStringList nameFilters;
    nameFilters << iFile.replace(".image.", ".").section(".", 0, -2) + ".mask.*.png";
    QDir currentDir(currentlyOpenedDir + iDir);
    currentDir.setFilter(QDir::Files);
    currentDir.setNameFilters(nameFilters);
    currentDir.setSorting(QDir::Name);
    QStringList files = currentDir.entryList();
    qSort(files.begin(), files.end(), maskFileLessThan);

    _current_obj_file_collection.clear();
    for (int i = 0 ; i<files.size()  ;++i)
    {
        std::string s(files[i].toLocal8Bit());
        for (int j = 1 ; j< MASK_TPYE_NUM ; ++j)
        {
            std::string target = S_mask_types[j];
            if (s.find(target))
            {
                _current_obj_file_collection[j] = files[i];
                break;
            }
        }
    }

    return _current_obj_file_collection;
}

