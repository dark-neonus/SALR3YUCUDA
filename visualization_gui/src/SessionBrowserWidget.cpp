/*
 * SessionBrowserWidget.cpp - Session list browser implementation
 */

#include "SessionBrowserWidget.h"
#include "DatabaseWrapper.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QHeaderView>
#include <QMenu>
#include <QInputDialog>
#include <QMessageBox>
#include <QClipboard>
#include <QApplication>

namespace salr {

SessionBrowserWidget::SessionBrowserWidget(DatabaseWrapper* database, QWidget* parent)
    : QWidget(parent)
    , database_(database)
{
    setupUi();
}

void SessionBrowserWidget::setupUi()
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    // Header
    QLabel* titleLabel = new QLabel(tr("Sessions"));
    titleLabel->setStyleSheet("font-weight: bold;");
    layout->addWidget(titleLabel);

    // Search/filter controls
    QHBoxLayout* filterLayout = new QHBoxLayout();
    filterLayout->setSpacing(4);

    searchEdit_ = new QLineEdit();
    searchEdit_->setPlaceholderText(tr("Search..."));
    searchEdit_->setClearButtonEnabled(true);
    filterLayout->addWidget(searchEdit_, 1);

    convergedCheck_ = new QCheckBox(tr("Converged"));
    filterLayout->addWidget(convergedCheck_);

    layout->addLayout(filterLayout);

    // Tree widget
    treeWidget_ = new QTreeWidget();
    treeWidget_->setHeaderLabels({
        tr("Name/ID"),
        tr("T"),
        tr("rho1"),
        tr("rho2"),
        tr("Grid"),
        tr("Snaps"),
        tr("Status")
    });
    treeWidget_->setRootIsDecorated(false);
    treeWidget_->setAlternatingRowColors(true);
    treeWidget_->setSortingEnabled(true);
    treeWidget_->setContextMenuPolicy(Qt::CustomContextMenu);
    treeWidget_->setSelectionMode(QAbstractItemView::SingleSelection);

    // Column widths
    QHeaderView* header = treeWidget_->header();
    header->setStretchLastSection(false);
    header->setSectionResizeMode(0, QHeaderView::Stretch);
    header->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(6, QHeaderView::ResizeToContents);

    layout->addWidget(treeWidget_, 1);

    // Connections
    connect(searchEdit_, &QLineEdit::textChanged,
            this, &SessionBrowserWidget::onFilterChanged);
    connect(convergedCheck_, &QCheckBox::toggled,
            this, &SessionBrowserWidget::onFilterChanged);
    connect(treeWidget_, &QTreeWidget::itemSelectionChanged,
            this, &SessionBrowserWidget::onItemSelectionChanged);
    connect(treeWidget_, &QTreeWidget::itemDoubleClicked,
            this, &SessionBrowserWidget::onItemDoubleClicked);
    connect(treeWidget_, &QTreeWidget::customContextMenuRequested,
            this, &SessionBrowserWidget::onContextMenu);
}

void SessionBrowserWidget::refresh()
{
    SessionFilter filter = buildFilter();
    QVector<SessionInfo> sessions = database_->listSessions(filter);
    populateList(sessions);
}

QString SessionBrowserWidget::selectedRunId() const
{
    QList<QTreeWidgetItem*> selected = treeWidget_->selectedItems();
    if (selected.isEmpty()) {
        return QString();
    }
    return selected.first()->data(0, Qt::UserRole).toString();
}

void SessionBrowserWidget::populateList(const QVector<SessionInfo>& sessions)
{
    QString previousSelection = selectedRunId();

    treeWidget_->clear();

    QString searchText = searchEdit_->text().toLower();

    for (const SessionInfo& session : sessions) {
        // Filter by search text
        if (!searchText.isEmpty()) {
            bool match = session.runId.toLower().contains(searchText) ||
                         session.nickname.toLower().contains(searchText);
            if (!match) continue;
        }

        QTreeWidgetItem* item = new QTreeWidgetItem();

        // Name/ID column - show nickname if set, otherwise run ID
        QString displayName = session.nickname.isEmpty() ?
            session.runId : session.nickname;
        item->setText(0, displayName);
        item->setToolTip(0, session.runId);

        // Temperature
        item->setText(1, QString::number(session.temperature, 'f', 2));

        // Densities
        item->setText(2, QString::number(session.rho1Bulk, 'f', 2));
        item->setText(3, QString::number(session.rho2Bulk, 'f', 2));

        // Grid size
        item->setText(4, QString("%1x%2").arg(session.nx).arg(session.ny));

        // Snapshot count
        item->setText(5, QString::number(session.snapshotCount));

        // Convergence status
        if (session.converged) {
            item->setText(6, tr("OK"));
            item->setForeground(6, QBrush(Qt::darkGreen));
        } else if (session.snapshotCount > 0) {
            item->setText(6, tr("..."));
            item->setForeground(6, QBrush(Qt::darkYellow));
        } else {
            item->setText(6, tr("-"));
            item->setForeground(6, QBrush(Qt::gray));
        }

        // Store run ID in user data
        item->setData(0, Qt::UserRole, session.runId);

        treeWidget_->addTopLevelItem(item);

        // Restore selection
        if (session.runId == previousSelection) {
            item->setSelected(true);
        }
    }

    // Sort by date (assuming run IDs contain timestamps)
    treeWidget_->sortItems(0, Qt::DescendingOrder);
}

SessionFilter SessionBrowserWidget::buildFilter() const
{
    SessionFilter filter;
    filter.convergedOnly = convergedCheck_->isChecked();
    return filter;
}

void SessionBrowserWidget::onItemSelectionChanged()
{
    QString runId = selectedRunId();
    if (!runId.isEmpty()) {
        emit sessionSelected(runId);
    }
}

void SessionBrowserWidget::onItemDoubleClicked(QTreeWidgetItem* item, int column)
{
    Q_UNUSED(column)
    if (item) {
        QString runId = item->data(0, Qt::UserRole).toString();
        emit sessionDoubleClicked(runId);
    }
}

void SessionBrowserWidget::onFilterChanged()
{
    refresh();
}

void SessionBrowserWidget::onContextMenu(const QPoint& pos)
{
    QTreeWidgetItem* item = treeWidget_->itemAt(pos);
    if (!item) return;

    QMenu menu(this);

    QAction* renameAction = menu.addAction(tr("Rename..."));
    connect(renameAction, &QAction::triggered, this, &SessionBrowserWidget::onRenameSession);

    QAction* copyAction = menu.addAction(tr("Copy Run ID"));
    connect(copyAction, &QAction::triggered, this, &SessionBrowserWidget::onCopyRunId);

    menu.addSeparator();

    QAction* deleteAction = menu.addAction(tr("Delete Session"));
    deleteAction->setIcon(QIcon::fromTheme("edit-delete"));
    connect(deleteAction, &QAction::triggered, this, &SessionBrowserWidget::onDeleteSession);

    menu.exec(treeWidget_->viewport()->mapToGlobal(pos));
}

void SessionBrowserWidget::onRenameSession()
{
    QString runId = selectedRunId();
    if (runId.isEmpty()) return;

    SessionInfo info = database_->getSessionInfo(runId);

    bool ok;
    QString nickname = QInputDialog::getText(this,
        tr("Rename Session"),
        tr("Enter nickname for session:"),
        QLineEdit::Normal,
        info.nickname,
        &ok);

    if (ok) {
        database_->setNickname(runId, nickname);
        refresh();
    }
}

void SessionBrowserWidget::onDeleteSession()
{
    QString runId = selectedRunId();
    if (runId.isEmpty()) return;

    int ret = QMessageBox::warning(this,
        tr("Delete Session"),
        tr("Are you sure you want to delete session '%1'?\n\n"
           "This will permanently delete all snapshots in this session.")
           .arg(runId),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (ret == QMessageBox::Yes) {
        if (database_->deleteSession(runId)) {
            refresh();
        }
    }
}

void SessionBrowserWidget::onCopyRunId()
{
    QString runId = selectedRunId();
    if (!runId.isEmpty()) {
        QApplication::clipboard()->setText(runId);
    }
}

} // namespace salr
