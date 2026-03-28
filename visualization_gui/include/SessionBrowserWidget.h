/*
 * SessionBrowserWidget.h - Session list browser widget
 */

#ifndef SALR_GUI_SESSION_BROWSER_WIDGET_H
#define SALR_GUI_SESSION_BROWSER_WIDGET_H

#include <QWidget>
#include <QTreeWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include "Types.h"

namespace salr {

class DatabaseWrapper;

class SessionBrowserWidget : public QWidget {
    Q_OBJECT

public:
    explicit SessionBrowserWidget(DatabaseWrapper* database, QWidget* parent = nullptr);

    void refresh();
    QString selectedRunId() const;

signals:
    void sessionSelected(const QString& runId);
    void sessionDoubleClicked(const QString& runId);

private slots:
    void onItemSelectionChanged();
    void onItemDoubleClicked(QTreeWidgetItem* item, int column);
    void onFilterChanged();
    void onContextMenu(const QPoint& pos);
    void onRenameSession();
    void onDeleteSession();
    void onCopyRunId();

private:
    void setupUi();
    void populateList(const QVector<SessionInfo>& sessions);
    SessionFilter buildFilter() const;

    DatabaseWrapper* database_;

    QLineEdit* searchEdit_ = nullptr;
    QComboBox* filterCombo_ = nullptr;
    QCheckBox* convergedCheck_ = nullptr;
    QTreeWidget* treeWidget_ = nullptr;
};

} // namespace salr

#endif // SALR_GUI_SESSION_BROWSER_WIDGET_H
