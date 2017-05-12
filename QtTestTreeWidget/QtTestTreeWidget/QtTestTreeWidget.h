#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtTestTreeWidget.h"

class QtTestTreeWidget : public QMainWindow
{
    Q_OBJECT

public:
    QtTestTreeWidget(QWidget *parent = Q_NULLPTR);

private:
    Ui::QtTestTreeWidgetClass ui;
};
