#include "text_editer_module.h"

#include <iostream>

#include <QAction>
#include <QPlainTextEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QFileDialog>
#include <QTextStream>
#include <QFile>
#include <QIODevice>


#include "my_main_window.h"


TextEditerModule::TextEditerModule(MyMainWindow*pMainWindow,
    QAction* pActionNew , 
    QAction* pActionOpen , 
    QAction* pActionSave,
    QAction* pActionSaveAs , 
    QAction* pActionUndo,
    QAction* pActionRedo,
    QAction* pActionCut,
    QAction* pActionCopy,
    QAction* pActionPaste,
    QPlainTextEdit* pTextEdit
    ,QObject* parent ):QObject(parent)
{
    m_pMainWindow = pMainWindow;
    m_pActionNew = pActionNew;
    m_pActionOpen = pActionOpen;
    m_pActionSave = pActionSave;
    m_pActionSaveAs = pActionSaveAs;
    m_pActionUndo = pActionUndo;
    m_pActionRedo = pActionRedo;
    m_pActionCut = pActionCut;
    m_pActionCopy = pActionCopy;
    m_pActionPaste = pActionPaste;
    m_pPlainTextEdit = pTextEdit;

    //Set window title & plain text edit initial status
    ResetTitile_i(true);

    connect(m_pActionNew , SIGNAL(triggered()) , this , SLOT(SlotActionNew()));
    connect(m_pActionOpen , SIGNAL(triggered()) , this , SLOT(SlotActionOpen()));
    connect(m_pActionSave , SIGNAL(triggered()) , this , SLOT(SlotActionSave()));
    connect(m_pActionSaveAs , SIGNAL(triggered()) , this , SLOT(SlotActionSaveAs()));
    connect(m_pActionUndo , SIGNAL(triggered()) , this , SLOT(SlotActionUndo()));
    connect(m_pActionRedo , SIGNAL(triggered()) , this , SLOT(SlotActionRedo()));
    connect(m_pActionCut , SIGNAL(triggered()) , this , SLOT(SlotActionCut()));
    connect(m_pActionCopy , SIGNAL(triggered()) , this , SLOT(SlotActionCopy()));
    connect(m_pActionPaste , SIGNAL(triggered()) , this , SLOT(SlotActionPaste()));

    connect(m_pPlainTextEdit , SIGNAL(textChanged()) , this, SLOT(SlotDocumentWasModified()));
}

TextEditerModule::~TextEditerModule()
{

}

void TextEditerModule::SlotActionNew()
{
    NewFile_i();
}

void TextEditerModule::SlotActionOpen()
{
    OpenFile_i();
}

void TextEditerModule::SlotActionSave()
{
    Save_i();
}

void TextEditerModule::SlotActionSaveAs()
{
    SaveAs_i();
}

void TextEditerModule::OpenFile_i()
{
    QString sCurFile = QFileDialog::getOpenFileName(m_pMainWindow , tr("Open file") ,tr("./") , tr("text (*.txt)"));
    if (!sCurFile.isEmpty())
    {
        QFile qFile(sCurFile);
        if (qFile.open(QFile::ReadOnly | QFile::Text))
        {
            QTextStream in(&qFile);
            QApplication::setOverrideCursor(Qt::WaitCursor);
            m_pPlainTextEdit->setPlainText(in.readAll());
            m_sCurFileName = QFileInfo(sCurFile).canonicalFilePath();
            ResetTitile_i(false);
            QApplication::restoreOverrideCursor();
        }
        else
        {
            QMessageBox::warning(m_pMainWindow , tr("Open File") , tr("Open file %1 failed!").arg(sCurFile));
        }
    }
    else
    {
        //QMessageBox::warning(m_pMainWindow , tr("Open File") , tr("File is empty!"));
    }
}

void TextEditerModule::NewFile_i()
{
    if (MaybeSave_i())
    {
        ResetTitile_i(true);
    }
}

bool TextEditerModule::SaveFile_i(const QString& file_name)
{
    QFile qFile(file_name);
    if (!qFile.open(QFile::WriteOnly  | QFile::Text))
    {
        QMessageBox::warning(m_pMainWindow , tr("Save file") , tr("Cant open file : %1").arg(file_name));
        return false;
    }
    else
    {
        QTextStream out(&qFile);
        out << m_pPlainTextEdit->toPlainText();

        QApplication::setOverrideCursor(Qt::WaitCursor);
        ResetTitile_i(false);
        QApplication::restoreOverrideCursor();
        return true;
    }
}


bool TextEditerModule::Save_i()
{
    if (m_bIsUntitled)
    {
        return SaveAs_i();
    }
    else
    {
        return SaveFile_i(m_sCurFileName);
    }
}

bool TextEditerModule::SaveAs_i()
{
    QString sChooseFile = QFileDialog::getSaveFileName(m_pMainWindow, tr("Save As") , m_sCurFileName , tr("text(*.txt)"));
    if (!sChooseFile.isEmpty())
    {
        return SaveFile_i(sChooseFile);
    }
    else
    {
        return false;
    }
}


bool TextEditerModule::MaybeSave_i()
{
    if (m_pPlainTextEdit->document()->isModified())
    {
        QMessageBox box;
        box.setWindowTitle("Warning");
        box.setIcon(QMessageBox::Warning);
        box.setText(m_sCurFileName + " save or not?");
        QPushButton* pYesBtn = new QPushButton(tr("Yes"));
        box.addButton(pYesBtn , QMessageBox::YesRole);
        QPushButton* pNoBtn = new QPushButton(tr("No"));
        box.addButton(pNoBtn , QMessageBox::NoRole);
        QPushButton* pCancelBtn = new QPushButton(tr("Cancel"));
        box.addButton(pCancelBtn , QMessageBox::RejectRole);
        box.exec();

        if (box.clickedButton()  == pYesBtn)
        {
            return Save_i();
        }
        else if(box.clickedButton() == pCancelBtn)
        {
            return false;
        }
        else if (box.clickedButton() == pNoBtn)
        {
            
        }

        return true;
    }
    else
    {
        return true;
    }
}

void TextEditerModule::SlotDocumentWasModified()
{
    m_pMainWindow->setWindowModified(m_pPlainTextEdit->document()->isModified());
}

void TextEditerModule::SetWindowTitle_i( const QString& sTitle )
{
    m_pMainWindow->setWindowTitle(sTitle + tr("[*]"));
}

void TextEditerModule::ResetTitile_i(bool bUntitled)
{
    m_bIsUntitled = bUntitled;
    if (bUntitled)
    {
        m_sCurFileName = tr("Untitled.txt");
        m_pPlainTextEdit->clear();
    }
    m_pMainWindow->setWindowTitle(m_sCurFileName + tr("[*]"));
    m_pPlainTextEdit->document()->setModified(false);
    m_pMainWindow->setWindowModified(false);
}

void TextEditerModule::SlotActionUndo()
{
    Undo_i();
}

void TextEditerModule::SlotActionRedo()
{
    Redo_i();
}

void TextEditerModule::SlotActionCut()
{
    Cut_i();
}

void TextEditerModule::SlotActionCopy()
{
    Copy_i();
}

void TextEditerModule::SlotActionPaste()
{
    Paste_i();
}

void TextEditerModule::Undo_i()
{
    m_pPlainTextEdit->undo();
}

void TextEditerModule::Redo_i()
{
    m_pPlainTextEdit->redo();
}

void TextEditerModule::Cut_i()
{
    m_pPlainTextEdit->cut();
}

void TextEditerModule::Copy_i()
{
    m_pPlainTextEdit->copy();
}

void TextEditerModule::Paste_i()
{
    m_pPlainTextEdit->paste();
}

bool TextEditerModule::CloseWindow()
{
    return MaybeSave_i();
}





