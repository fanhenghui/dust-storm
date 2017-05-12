#include "QtTestTreeWidget.h"

#include <QCheckBox>

#include <iostream>
#include <string>

namespace
{
    void add_child_check_box(QTreeWidget *tree_widget , QTreeWidgetItem* tree_widget_item)
    {
        if (0 == tree_widget_item->childCount())
        {
            QCheckBox* check_box = new QCheckBox();
            check_box->setText(QString(""));
            tree_widget->setItemWidget(tree_widget_item, 1, check_box);
        }
        else
        {
            for (int i = 0 ;i<tree_widget_item->childCount() ; ++i)
            {
                add_child_check_box(tree_widget , tree_widget_item->child(i));
            }
        }
    }
}

QtTestTreeWidget::QtTestTreeWidget(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    std::cout << "column count : " << ui.PathologyTreeWidget->columnCount();

    int top_level_item_count = ui.PathologyTreeWidget->topLevelItemCount();
    std::cout << "top level item count : " << top_level_item_count;

    //Add check box to each child
    for (int i = 0; i<top_level_item_count ; ++i)
    {
        QTreeWidgetItem* treeItem = ui.PathologyTreeWidget->topLevelItem(i);
        add_child_check_box(ui.PathologyTreeWidget , treeItem);
    }
}
