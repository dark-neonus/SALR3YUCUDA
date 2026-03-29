/*
 * SnapshotBrowserWidget.h - Snapshot thumbnail browser
 */

#ifndef SALR_GUI_SNAPSHOT_BROWSER_WIDGET_H
#define SALR_GUI_SNAPSHOT_BROWSER_WIDGET_H

#include <QWidget>
#include <QListWidget>
#include <QLabel>
#include <QMap>
#include <QPixmap>
#include "Types.h"

namespace salr {

class DatabaseWrapper;

class SnapshotBrowserWidget : public QWidget {
    Q_OBJECT

public:
    explicit SnapshotBrowserWidget(DatabaseWrapper* database, QWidget* parent = nullptr);

    void setSession(const QString& runId);
    void refresh();
    int selectedIteration() const;

signals:
    void snapshotSelected(int iteration);
    void snapshotDoubleClicked(int iteration);

private slots:
    void onItemSelectionChanged();
    void onItemDoubleClicked(QListWidgetItem* item);
    void onContextMenu(const QPoint& pos);
    void onExportSnapshot();
    void onDeleteSnapshot();
    void onBranchFromSnapshot();

private:
    void setupUi();
    void loadSnapshots();
    QImage generateThumbnail(const SnapshotData& data, int size = 64);
    void loadThumbnailAsync(int iteration);

    DatabaseWrapper* database_;
    QString currentRunId_;

    QLabel* titleLabel_ = nullptr;
    QListWidget* listWidget_ = nullptr;

    // Thumbnail cache
    QMap<QString, QPixmap> thumbnailCache_;
};

} // namespace salr

#endif // SALR_GUI_SNAPSHOT_BROWSER_WIDGET_H
