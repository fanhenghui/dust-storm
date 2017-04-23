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


TextEditerModule::TextEditerModule(MyMainWindow*main_window,
    QAction* action_new , 
    QAction* action_open , 
    QAction* action_save,
    QAction* action_save_as , 
    QAction* action_undo,
    QAction* action_redo,
    QAction* action_cut,
    QAction* action_copy,
    QAction* action_paste,
    QPlainTextEdit* text_edit
    ,QObject* parent ):QObject(parent)
{
    _main_window = main_window;
    _action_new = action_new;
    _action_open = action_open;
    action_save = action_save;
    action_save_as = action_save_as;
    action_undo = action_undo;
    _action_redo = action_redo;
    _action_cut = action_cut;
    _action_copy = action_copy;
    _action_paste = action_paste;
    _plain_text_rdit = text_edit;

    //Set window title & plain text edit initial status
    reset_titile_i(true);

    connect(_action_new , SIGNAL(triggered()) , this , SLOT(slot_action_new()));
    connect(_action_open , SIGNAL(triggered()) , this , SLOT(slot_action_open()));
    connect(action_save , SIGNAL(triggered()) , this , SLOT(slot_action_save()));
    connect(action_save_as , SIGNAL(triggered()) , this , SLOT(slot_action_save_as()));
    connect(action_undo , SIGNAL(triggered()) , this , SLOT(slot_action_undo()));
    connect(_action_redo , SIGNAL(triggered()) , this , SLOT(slot_action_redo()));
    connect(_action_cut , SIGNAL(triggered()) , this , SLOT(slot_action_cut()));
    connect(_action_copy , SIGNAL(triggered()) , this , SLOT(slot_action_copy()));
    connect(_action_paste , SIGNAL(triggered()) , this , SLOT(slot_action_paste()));

    connect(_plain_text_rdit , SIGNAL(textChanged()) , this, SLOT(slot_document_was_modified()));
}

TextEditerModule::~TextEditerModule()
{

}

void TextEditerModule::slot_action_new()
{
    new_file_i();
}

void TextEditerModule::slot_action_open()
{
    open_file_i();
}

void TextEditerModule::slot_action_save()
{
    save_i();
}

void TextEditerModule::slot_action_save_as()
{
    save_as_i();
}

void TextEditerModule::open_file_i()
{
    QString file_name = QFileDialog::getOpenFileName(_main_window , tr("Open file") ,tr("./") , tr("text (*.txt)"));
    if (!file_name.isEmpty())
    {
        QFile q_file(file_name);
        if (q_file.open(QFile::ReadOnly | QFile::Text))
        {
            QTextStream in(&q_file);
            QApplication::setOverrideCursor(Qt::WaitCursor);
            _plain_text_rdit->setPlainText(in.readAll());
            _cur_file_name = QFileInfo(file_name).canonicalFilePath();
            reset_titile_i(false);
            QApplication::restoreOverrideCursor();
        }
        else
        {
            QMessageBox::warning(_main_window , tr("Open File") , tr("Open file %1 failed!").arg(file_name));
        }
    }
    else
    {
        //QMessageBox::warning(_main_window , tr("Open File") , tr("File is empty!"));
    }
}

void TextEditerModule::new_file_i()
{
    if (maybe_save_i())
    {
        reset_titile_i(true);
    }
}

bool TextEditerModule::save_file_i(const QString& file_name)
{
    QFile q_file(file_name);
    if (!q_file.open(QFile::WriteOnly  | QFile::Text))
    {
        QMessageBox::warning(_main_window , tr("Save file") , tr("Cant open file : %1").arg(file_name));
        return false;
    }
    else
    {
        QTextStream out(&q_file);
        out << _plain_text_rdit->toPlainText();

        QApplication::setOverrideCursor(Qt::WaitCursor);
        reset_titile_i(false);
        QApplication::restoreOverrideCursor();
        return true;
    }
}


bool TextEditerModule::save_i()
{
    if (_is_untitled)
    {
        return save_as_i();
    }
    else
    {
        return save_file_i(_cur_file_name);
    }
}

bool TextEditerModule::save_as_i()
{
    QString file_name = QFileDialog::getSaveFileName(_main_window, tr("Save As") , _cur_file_name , tr("text(*.txt)"));
    if (!file_name.isEmpty())
    {
        return save_file_i(file_name);
    }
    else
    {
        return false;
    }
}


bool TextEditerModule::maybe_save_i()
{
    if (_plain_text_rdit->document()->isModified())
    {
        QMessageBox box;
        box.setWindowTitle("Warning");
        box.setIcon(QMessageBox::Warning);
        box.setText(_cur_file_name + " save or not?");
        QPushButton* yes_btn = new QPushButton(tr("Yes"));
        box.addButton(yes_btn , QMessageBox::YesRole);
        QPushButton* no_btn = new QPushButton(tr("No"));
        box.addButton(no_btn , QMessageBox::NoRole);
        QPushButton* cancel_btn = new QPushButton(tr("Cancel"));
        box.addButton(cancel_btn , QMessageBox::RejectRole);
        box.exec();

        if (box.clickedButton()  == yes_btn)
        {
            return save_i();
        }
        else if(box.clickedButton() == cancel_btn)
        {
            return false;
        }
        else if (box.clickedButton() == no_btn)
        {
            
        }

        return true;
    }
    else
    {
        return true;
    }
}

void TextEditerModule::slot_document_was_modified()
{
    _main_window->setWindowModified(_plain_text_rdit->document()->isModified());
}

void TextEditerModule::set_window_title_i( const QString& title )
{
    _main_window->setWindowTitle(title + tr("[*]"));
}

void TextEditerModule::reset_titile_i(bool is_untitled)
{
    _is_untitled = is_untitled;
    if (is_untitled)
    {
        _cur_file_name = tr("Untitled.txt");
        _plain_text_rdit->clear();
    }
    _main_window->setWindowTitle(_cur_file_name + tr("[*]"));
    _plain_text_rdit->document()->setModified(false);
    _main_window->setWindowModified(false);
}

void TextEditerModule::slot_action_undo()
{
    undo_i();
}

void TextEditerModule::slot_action_redo()
{
    redo_i();
}

void TextEditerModule::slot_action_cut()
{
    cut_i();
}

void TextEditerModule::slot_action_copy()
{
    copy_i();
}

void TextEditerModule::slot_action_paste()
{
    paste_i();
}

void TextEditerModule::undo_i()
{
    _plain_text_rdit->undo();
}

void TextEditerModule::redo_i()
{
    _plain_text_rdit->redo();
}

void TextEditerModule::cut_i()
{
    _plain_text_rdit->cut();
}

void TextEditerModule::copy_i()
{
    _plain_text_rdit->copy();
}

void TextEditerModule::paste_i()
{
    _plain_text_rdit->paste();
}

bool TextEditerModule::close_window()
{
    return maybe_save_i();
}





