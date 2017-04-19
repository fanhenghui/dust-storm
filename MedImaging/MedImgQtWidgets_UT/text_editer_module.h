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
        MyMainWindow*pMainWindow , 
        QAction* pActionNew , 
        QAction* pActionOpen , 
        QAction* pActionSave, 
        QAction* pActionSaveAs , 
        QAction* pActionUndo,
        QAction* pActionRedo,
        QAction* pActionCut,
        QAction* pActionCopy,
        QAction* pActionPaste,
        QPlainTextEdit* pTextEdit ,
        QObject* parent = nullptr );
    ~TextEditerModule();

    bool CloseWindow();

protected:

private slots:
    void SlotActionNew();
    void SlotActionOpen();
    void SlotActionSave();
    void SlotActionSaveAs();
    void SlotActionUndo();
    void SlotActionRedo();
    void SlotActionCut();
    void SlotActionCopy();
    void SlotActionPaste();
    void SlotDocumentWasModified();

private:
    void OpenFile_i();

    void NewFile_i();

    bool SaveFile_i(const QString& sFile);

    bool Save_i();

    bool SaveAs_i();

    bool MaybeSave_i();

    void Undo_i();

    void Redo_i();

    void Cut_i();

    void Copy_i();

    void Paste_i();

    void SetWindowTitle_i(const QString& sTitle);

    void ResetTitile_i(bool bUntitled );


private:
    MyMainWindow* m_pMainWindow;
    QAction *m_pActionNew;
    QAction *m_pActionOpen;
    QAction *m_pActionSave;
    QAction *m_pActionSaveAs;
    QAction* m_pActionUndo;
    QAction* m_pActionRedo;
    QAction* m_pActionCut;
    QAction* m_pActionCopy;
    QAction* m_pActionPaste;
    QPlainTextEdit *m_pPlainTextEdit;

    QString m_sCurFileName;
    bool m_bIsUntitled;
};

#endif