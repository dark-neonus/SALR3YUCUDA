/*
 * SnapshotSelectorDialog.cpp - Snapshot selection dialog implementation
 */

#include "SnapshotSelectorDialog.h"
#include "DatabaseWrapper.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QDialogButtonBox>
#include <QPushButton>

namespace salr {

SnapshotSelectorDialog::SnapshotSelectorDialog(DatabaseWrapper* database, QWidget* parent)
    : QDialog(parent)
    , database_(database)
{
    setWindowTitle(tr("Select Snapshot"));
    setMinimumSize(600, 400);
    setupUi();
    loadSessions();
}

void SnapshotSelectorDialog::setupUi()
{
    QVBoxLayout* layout = new QVBoxLayout(this);

    QSplitter* splitter = new QSplitter(Qt::Horizontal);

    // Session list
    QGroupBox* sessionGroup = new QGroupBox(tr("Sessions"));
    QVBoxLayout* sessionLayout = new QVBoxLayout(sessionGroup);
    sessionList_ = new QListWidget();
    sessionLayout->addWidget(sessionList_);
    splitter->addWidget(sessionGroup);

    // Snapshot list
    QGroupBox* snapshotGroup = new QGroupBox(tr("Snapshots"));
    QVBoxLayout* snapshotLayout = new QVBoxLayout(snapshotGroup);
    snapshotList_ = new QListWidget();
    snapshotLayout->addWidget(snapshotList_);
    splitter->addWidget(snapshotGroup);

    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);

    layout->addWidget(splitter, 1);

    // Info label
    infoLabel_ = new QLabel();
    layout->addWidget(infoLabel_);

    // Dialog buttons
    QDialogButtonBox* buttonBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &SnapshotSelectorDialog::onAccept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttonBox);

    // Connections
    connect(sessionList_, &QListWidget::itemSelectionChanged,
            this, &SnapshotSelectorDialog::onSessionSelected);
    connect(snapshotList_, &QListWidget::itemSelectionChanged,
            this, &SnapshotSelectorDialog::onSnapshotSelected);
    connect(snapshotList_, &QListWidget::itemDoubleClicked, this, [this]() {
        if (selectedIteration_ >= 0) {
            accept();
        }
    });
}

void SnapshotSelectorDialog::setSession(const QString& runId)
{
    // Find and select the session
    for (int i = 0; i < sessionList_->count(); ++i) {
        if (sessionList_->item(i)->data(Qt::UserRole).toString() == runId) {
            sessionList_->setCurrentRow(i);
            break;
        }
    }
}

void SnapshotSelectorDialog::loadSessions()
{
    sessionList_->clear();

    if (!database_->isInitialized()) {
        return;
    }

    QVector<SessionInfo> sessions = database_->listSessions();

    for (const SessionInfo& session : sessions) {
        QString displayName = session.nickname.isEmpty() ?
            session.runId : QString("%1 (%2)").arg(session.nickname).arg(session.runId);

        QListWidgetItem* item = new QListWidgetItem(displayName);
        item->setData(Qt::UserRole, session.runId);
        item->setToolTip(QString("T=%1, rho1=%2, rho2=%3, %4x%5")
            .arg(session.temperature, 0, 'f', 2)
            .arg(session.rho1Bulk, 0, 'f', 2)
            .arg(session.rho2Bulk, 0, 'f', 2)
            .arg(session.nx)
            .arg(session.ny));

        sessionList_->addItem(item);
    }
}

void SnapshotSelectorDialog::loadSnapshots(const QString& runId)
{
    snapshotList_->clear();

    if (runId.isEmpty()) {
        return;
    }

    QVector<int> snapshots = database_->listSnapshots(runId);

    for (int iter : snapshots) {
        QListWidgetItem* item = new QListWidgetItem(QString("Iteration %1").arg(iter));
        item->setData(Qt::UserRole, iter);
        snapshotList_->addItem(item);
    }

    // Select latest by default
    if (snapshotList_->count() > 0) {
        snapshotList_->setCurrentRow(snapshotList_->count() - 1);
    }
}

void SnapshotSelectorDialog::onSessionSelected()
{
    QList<QListWidgetItem*> selected = sessionList_->selectedItems();
    if (selected.isEmpty()) {
        snapshotList_->clear();
        selectedRunId_.clear();
        return;
    }

    selectedRunId_ = selected.first()->data(Qt::UserRole).toString();
    loadSnapshots(selectedRunId_);
}

void SnapshotSelectorDialog::onSnapshotSelected()
{
    QList<QListWidgetItem*> selected = snapshotList_->selectedItems();

    QDialogButtonBox* buttonBox = findChild<QDialogButtonBox*>();

    if (selected.isEmpty()) {
        selectedIteration_ = -1;
        infoLabel_->clear();
        if (buttonBox) buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        return;
    }

    selectedIteration_ = selected.first()->data(Qt::UserRole).toInt();

    // Load and display metadata
    SnapshotMeta meta = database_->loadMetadata(selectedRunId_, selectedIteration_);

    infoLabel_->setText(QString("Iteration %1: error=%2, %3x%4 grid")
        .arg(meta.iteration)
        .arg(meta.currentError, 0, 'e', 2)
        .arg(meta.nx)
        .arg(meta.ny));

    if (buttonBox) buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
}

void SnapshotSelectorDialog::onAccept()
{
    if (!selectedRunId_.isEmpty() && selectedIteration_ >= 0) {
        accept();
    }
}

} // namespace salr
