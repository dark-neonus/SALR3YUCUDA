/*
 * ParameterDisplayWidget.cpp - Hierarchical parameter display implementation
 */

#include "ParameterDisplayWidget.h"

#include <QVBoxLayout>
#include <QHeaderView>

namespace salr {

ParameterDisplayWidget::ParameterDisplayWidget(QWidget* parent)
    : QWidget(parent)
{
    setupUi();
}

void ParameterDisplayWidget::setupUi()
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    treeWidget_ = new QTreeWidget();
    treeWidget_->setHeaderLabels({tr("Parameter"), tr("Value")});
    treeWidget_->setAlternatingRowColors(true);
    treeWidget_->setRootIsDecorated(true);
    treeWidget_->setIndentation(15);

    QHeaderView* header = treeWidget_->header();
    header->setStretchLastSection(true);
    header->setSectionResizeMode(0, QHeaderView::ResizeToContents);

    layout->addWidget(treeWidget_);
}

void ParameterDisplayWidget::setSessionInfo(const SessionInfo& info)
{
    currentSession_ = info;
    hasSession_ = true;
    updateDisplay();
}

void ParameterDisplayWidget::setSnapshotMeta(const SnapshotMeta& meta)
{
    currentSnapshot_ = meta;
    hasSnapshot_ = true;
    updateDisplay();
}

void ParameterDisplayWidget::clear()
{
    hasSession_ = false;
    hasSnapshot_ = false;
    treeWidget_->clear();
}

void ParameterDisplayWidget::updateDisplay()
{
    treeWidget_->clear();

    if (!hasSession_ && !hasSnapshot_) {
        return;
    }

    // Session info
    if (hasSession_) {
        QTreeWidgetItem* sessionGroup = addGroup(tr("Session"));

        addParam(sessionGroup, tr("Run ID"), currentSession_.runId);
        if (!currentSession_.nickname.isEmpty()) {
            addParam(sessionGroup, tr("Nickname"), currentSession_.nickname);
        }
        addParam(sessionGroup, tr("Created"), currentSession_.createdAt.toString(Qt::ISODate));
        addParam(sessionGroup, tr("Snapshots"), currentSession_.snapshotCount);
        addParam(sessionGroup, tr("Converged"), currentSession_.converged ? tr("Yes") : tr("No"));
        if (currentSession_.finalError >= 0) {
            addParam(sessionGroup, tr("Final Error"), currentSession_.finalError, 2);
        }
        if (!currentSession_.source.isEmpty()) {
            addParam(sessionGroup, tr("Branched From"), currentSession_.source);
        }

        sessionGroup->setExpanded(true);
    }

    // Grid parameters
    QTreeWidgetItem* gridGroup = addGroup(tr("Grid"));

    if (hasSnapshot_) {
        addParam(gridGroup, "nx", currentSnapshot_.nx);
        addParam(gridGroup, "ny", currentSnapshot_.ny);
        addParam(gridGroup, "dx", currentSnapshot_.dx, 3);
        addParam(gridGroup, "dy", currentSnapshot_.dy, 3);
        addParam(gridGroup, "Lx", currentSnapshot_.Lx, 2);
        addParam(gridGroup, "Ly", currentSnapshot_.Ly, 2);
        addParam(gridGroup, tr("Boundary"), currentSnapshot_.boundaryMode);
    } else if (hasSession_) {
        addParam(gridGroup, "nx", currentSession_.nx);
        addParam(gridGroup, "ny", currentSession_.ny);
        addParam(gridGroup, "dx", currentSession_.dx, 3);
        addParam(gridGroup, "dy", currentSession_.dy, 3);
        addParam(gridGroup, tr("Boundary"), currentSession_.boundaryMode);
    }

    gridGroup->setExpanded(true);

    // Physics parameters
    QTreeWidgetItem* physicsGroup = addGroup(tr("Physics"));

    if (hasSnapshot_) {
        addParam(physicsGroup, tr("Temperature"), currentSnapshot_.temperature, 3);
        addParam(physicsGroup, tr("rho1 (bulk)"), currentSnapshot_.rho1Bulk, 3);
        addParam(physicsGroup, tr("rho2 (bulk)"), currentSnapshot_.rho2Bulk, 3);
        addParam(physicsGroup, tr("Cutoff"), currentSnapshot_.cutoffRadius, 1);
    } else if (hasSession_) {
        addParam(physicsGroup, tr("Temperature"), currentSession_.temperature, 3);
        addParam(physicsGroup, tr("rho1 (bulk)"), currentSession_.rho1Bulk, 3);
        addParam(physicsGroup, tr("rho2 (bulk)"), currentSession_.rho2Bulk, 3);
    }

    physicsGroup->setExpanded(true);

    // Solver parameters (from snapshot)
    if (hasSnapshot_) {
        QTreeWidgetItem* solverGroup = addGroup(tr("Solver"));

        addParam(solverGroup, "xi1", currentSnapshot_.xi1, 6);
        addParam(solverGroup, "xi2", currentSnapshot_.xi2, 6);

        solverGroup->setExpanded(false);
    }

    // Snapshot state
    if (hasSnapshot_) {
        QTreeWidgetItem* stateGroup = addGroup(tr("Snapshot State"));

        addParam(stateGroup, tr("Iteration"), currentSnapshot_.iteration);
        addParam(stateGroup, tr("Error"), currentSnapshot_.currentError, 2);
        addParam(stateGroup, tr("Delta Error"), currentSnapshot_.deltaError, 2);
        addParam(stateGroup, tr("Timestamp"), currentSnapshot_.createdAt.toString(Qt::ISODate));

        stateGroup->setExpanded(true);
    }
}

QTreeWidgetItem* ParameterDisplayWidget::addGroup(const QString& name)
{
    QTreeWidgetItem* item = new QTreeWidgetItem(treeWidget_);
    item->setText(0, name);
    item->setFlags(item->flags() & ~Qt::ItemIsSelectable);

    QFont font = item->font(0);
    font.setBold(true);
    item->setFont(0, font);

    return item;
}

void ParameterDisplayWidget::addParam(QTreeWidgetItem* parent, const QString& name, const QString& value)
{
    QTreeWidgetItem* item = new QTreeWidgetItem(parent);
    item->setText(0, name);
    item->setText(1, value);
}

void ParameterDisplayWidget::addParam(QTreeWidgetItem* parent, const QString& name, double value, int precision)
{
    QString strValue;
    if (qAbs(value) < 1e-3 && value != 0) {
        strValue = QString::number(value, 'e', precision);
    } else {
        strValue = QString::number(value, 'g', precision + 2);
    }
    addParam(parent, name, strValue);
}

void ParameterDisplayWidget::addParam(QTreeWidgetItem* parent, const QString& name, int value)
{
    addParam(parent, name, QString::number(value));
}

} // namespace salr
