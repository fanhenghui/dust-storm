#ifndef MI_TEXT_EDITER_MODULE_H_
#define MI_TEXT_EDITER_MODULE_H_

#include <QObject>

class QAction;
class QPlainTextEdit;
class MyMainWindow;
class TextEditerModule : public QObject
{
    Q_OBJECT
public:
    TextEditerModule(
        MyMainWindow*main_window , 
        QAction* action_new , 
        QAction* action_open , 
        QAction* action_save, 
        QAction* action_save_as , 
        QAction* action_undo,
        QAction* action_redo,
        QAction* action_cut,
        QAction* action_copy,
        QAction* action_paste,
        QPlainTextEdit* text_edit ,
        QObject* parent = nullptr );
    ~TextEditerModule();

    bool close_window();

protected:

private slots:
    void slot_action_new();
    void slot_action_open();
    void slot_action_save();
    void slot_action_save_as();
    void slot_action_undo();
    void slot_action_redo();
    void slot_action_cut();
    void slot_action_copy();
    void slot_action_paste();
    void slot_document_was_modified();

private:
    void open_file_i();

    void new_file_i();

    bool save_file_i(const QString& file_name);

    bool save_i();

    bool save_as_i();

    bool maybe_save_i();

    void undo_i();

    void redo_i();

    void cut_i();

    void copy_i();

    void paste_i();

    void set_window_title_i(const QString& title);

    void reset_titile_i(bool is_untitled );


private:
    MyMainWindow* _main_window;
    QAction *_action_new;
    QAction *_action_open;
    QAction *_action_save;
    QAction *_action_save_as;
    QAction* _action_undo;
    QAction* _action_redo;
    QAction* _action_cut;
    QAction* _action_copy;
    QAction* _action_paste;
    QPlainTextEdit *_plain_text_rdit;

    QString _cur_file_name;
    bool _is_untitled;
};

#endif