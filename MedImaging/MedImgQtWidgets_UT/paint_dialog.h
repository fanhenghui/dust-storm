#ifndef PAINT_DIALOG_H
#define PAINT_DIALOG_H

#include "Qt/qdialog.h"

class PaintDialog : public QDialog
{
    Q_OBJECT
public:
    PaintDialog(QDialog* parent = 0);
    virtual ~PaintDialog();
protected:
    virtual void paintEvent(QPaintEvent *);
    virtual void mouseMoveEvent(QMouseEvent *);
    virtual void mousePressEvent(QMouseEvent *);
    virtual void mouseReleaseEvent(QMouseEvent *);
private:
    QPoint _pre_point;
    QPoint _cur_point;
};

#endif