/*
 * ParameterDisplayWidget.h - Hierarchical parameter display
 */

#ifndef SALR_GUI_PARAMETER_DISPLAY_WIDGET_H
#define SALR_GUI_PARAMETER_DISPLAY_WIDGET_H

#include <QWidget>
#include <QTreeWidget>
#include "Types.h"

namespace salr {

class ParameterDisplayWidget : public QWidget {
    Q_OBJECT

public:
    explicit ParameterDisplayWidget(QWidget* parent = nullptr);

    void setSessionInfo(const SessionInfo& info);
    void setSnapshotMeta(const SnapshotMeta& meta);
    void clear();

private:
    void setupUi();
    void updateDisplay();

    QTreeWidgetItem* addGroup(const QString& name);
    void addParam(QTreeWidgetItem* parent, const QString& name, const QString& value);
    void addParam(QTreeWidgetItem* parent, const QString& name, double value, int precision = 4);
    void addParam(QTreeWidgetItem* parent, const QString& name, int value);

    QTreeWidget* treeWidget_ = nullptr;

    SessionInfo currentSession_;
    SnapshotMeta currentSnapshot_;
    bool hasSession_ = false;
    bool hasSnapshot_ = false;
};

} // namespace salr

#endif // SALR_GUI_PARAMETER_DISPLAY_WIDGET_H
