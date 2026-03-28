/*
 * SnapshotSelectorDialog.h - Snapshot selection dialog for branching
 */

#ifndef SALR_GUI_SNAPSHOT_SELECTOR_DIALOG_H
#define SALR_GUI_SNAPSHOT_SELECTOR_DIALOG_H

#include <QDialog>
#include <QListWidget>
#include <QLabel>
#include "Types.h"

namespace salr {

class DatabaseWrapper;

class SnapshotSelectorDialog : public QDialog {
    Q_OBJECT

public:
    explicit SnapshotSelectorDialog(DatabaseWrapper* database, QWidget* parent = nullptr);

    void setSession(const QString& runId);

    QString selectedRunId() const { return selectedRunId_; }
    int selectedIteration() const { return selectedIteration_; }

private slots:
    void onSessionSelected();
    void onSnapshotSelected();
    void onAccept();

private:
    void setupUi();
    void loadSessions();
    void loadSnapshots(const QString& runId);

    DatabaseWrapper* database_;

    QListWidget* sessionList_ = nullptr;
    QListWidget* snapshotList_ = nullptr;
    QLabel* infoLabel_ = nullptr;

    QString selectedRunId_;
    int selectedIteration_ = -1;
};

} // namespace salr

#endif // SALR_GUI_SNAPSHOT_SELECTOR_DIALOG_H
