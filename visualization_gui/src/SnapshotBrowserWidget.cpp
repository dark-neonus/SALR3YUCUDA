/*
 * SnapshotBrowserWidget.cpp - Snapshot thumbnail browser implementation
 */

#include "SnapshotBrowserWidget.h"
#include "DatabaseWrapper.h"

#include <QVBoxLayout>
#include <QMenu>
#include <QFileDialog>
#include <QMessageBox>
#include <QPainter>
#include <QtConcurrent>
#include <QFutureWatcher>

namespace salr {

SnapshotBrowserWidget::SnapshotBrowserWidget(DatabaseWrapper* database, QWidget* parent)
    : QWidget(parent)
    , database_(database)
{
    setupUi();
}

void SnapshotBrowserWidget::setupUi()
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    titleLabel_ = new QLabel(tr("Snapshots"));
    titleLabel_->setStyleSheet("font-weight: bold;");
    layout->addWidget(titleLabel_);

    listWidget_ = new QListWidget();
    listWidget_->setViewMode(QListView::IconMode);
    listWidget_->setIconSize(QSize(64, 64));
    listWidget_->setSpacing(4);
    listWidget_->setResizeMode(QListView::Adjust);
    listWidget_->setMovement(QListView::Static);
    listWidget_->setContextMenuPolicy(Qt::CustomContextMenu);
    listWidget_->setSelectionMode(QAbstractItemView::SingleSelection);

    layout->addWidget(listWidget_, 1);

    // Connections
    connect(listWidget_, &QListWidget::itemSelectionChanged,
            this, &SnapshotBrowserWidget::onItemSelectionChanged);
    connect(listWidget_, &QListWidget::itemDoubleClicked,
            this, &SnapshotBrowserWidget::onItemDoubleClicked);
    connect(listWidget_, &QListWidget::customContextMenuRequested,
            this, &SnapshotBrowserWidget::onContextMenu);
}

void SnapshotBrowserWidget::setSession(const QString& runId)
{
    currentRunId_ = runId;
    loadSnapshots();
}

void SnapshotBrowserWidget::refresh()
{
    if (!currentRunId_.isEmpty()) {
        loadSnapshots();
    }
}

int SnapshotBrowserWidget::selectedIteration() const
{
    QList<QListWidgetItem*> selected = listWidget_->selectedItems();
    if (selected.isEmpty()) {
        return -1;
    }
    return selected.first()->data(Qt::UserRole).toInt();
}

void SnapshotBrowserWidget::loadSnapshots()
{
    listWidget_->clear();

    if (currentRunId_.isEmpty()) {
        titleLabel_->setText(tr("Snapshots"));
        return;
    }

    QVector<int> iterations = database_->listSnapshots(currentRunId_);

    titleLabel_->setText(tr("Snapshots (%1)").arg(iterations.size()));

    for (int iter : iterations) {
        QListWidgetItem* item = new QListWidgetItem();
        item->setText(QString("Iter %1").arg(iter));
        item->setData(Qt::UserRole, iter);

        // Check thumbnail cache
        QString cacheKey = QString("%1_%2").arg(currentRunId_).arg(iter);
        if (thumbnailCache_.contains(cacheKey)) {
            item->setIcon(QIcon(thumbnailCache_[cacheKey]));
        } else {
            // Create placeholder icon
            QPixmap placeholder(64, 64);
            placeholder.fill(Qt::lightGray);
            QPainter p(&placeholder);
            p.drawText(placeholder.rect(), Qt::AlignCenter, QString::number(iter));
            item->setIcon(QIcon(placeholder));

            // Queue async thumbnail loading
            loadThumbnailAsync(iter);
        }

        listWidget_->addItem(item);
    }
}

void SnapshotBrowserWidget::loadThumbnailAsync(int iteration)
{
    QString runId = currentRunId_;
    QString cacheKey = QString("%1_%2").arg(runId).arg(iteration);

    // Use QtConcurrent to load in background
    QFutureWatcher<QPixmap>* watcher = new QFutureWatcher<QPixmap>(this);

    connect(watcher, &QFutureWatcher<QPixmap>::finished, this, [this, watcher, cacheKey, iteration]() {
        QPixmap thumbnail = watcher->result();
        watcher->deleteLater();

        if (!thumbnail.isNull()) {
            thumbnailCache_[cacheKey] = thumbnail;

            // Update item if still visible
            for (int i = 0; i < listWidget_->count(); ++i) {
                QListWidgetItem* item = listWidget_->item(i);
                if (item->data(Qt::UserRole).toInt() == iteration) {
                    item->setIcon(QIcon(thumbnail));
                    break;
                }
            }
        }
    });

    QFuture<QPixmap> future = QtConcurrent::run([this, runId, iteration]() -> QPixmap {
        SnapshotData data = database_->loadSnapshot(runId, iteration);
        if (!data.isValid()) {
            return QPixmap();
        }
        return generateThumbnail(data);
    });

    watcher->setFuture(future);
}

QPixmap SnapshotBrowserWidget::generateThumbnail(const SnapshotData& data, int size)
{
    if (!data.isValid()) {
        return QPixmap();
    }

    int nx = data.meta.nx;
    int ny = data.meta.ny;

    // Create image
    QImage image(nx, ny, QImage::Format_RGB32);

    // Compute color scale (clipped to 3x mean)
    double rho1Max = 3.0 * data.rho1Mean;
    double rho2Max = 3.0 * data.rho2Mean;

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int idx = iy * nx + ix;

            double r1 = data.rho1[idx];
            double r2 = data.rho2[idx];

            // Normalize to [0, 1]
            double n1 = qBound(0.0, r1 / rho1Max, 1.0);
            double n2 = qBound(0.0, r2 / rho2Max, 1.0);

            // Map to colors: species 1 = purple (R=139, B=139), species 2 = green (G=139)
            int r = static_cast<int>(139 * n1);
            int g = static_cast<int>(139 * n2);
            int b = static_cast<int>(139 * n1);

            image.setPixel(ix, ny - 1 - iy, qRgb(r, g, b));  // Flip Y
        }
    }

    // Scale to thumbnail size
    QPixmap pixmap = QPixmap::fromImage(image);
    return pixmap.scaled(size, size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

void SnapshotBrowserWidget::onItemSelectionChanged()
{
    int iteration = selectedIteration();
    if (iteration >= 0) {
        emit snapshotSelected(iteration);
    }
}

void SnapshotBrowserWidget::onItemDoubleClicked(QListWidgetItem* item)
{
    if (item) {
        int iteration = item->data(Qt::UserRole).toInt();
        emit snapshotDoubleClicked(iteration);
    }
}

void SnapshotBrowserWidget::onContextMenu(const QPoint& pos)
{
    QListWidgetItem* item = listWidget_->itemAt(pos);
    if (!item) return;

    QMenu menu(this);

    QAction* exportAction = menu.addAction(tr("Export to ASCII..."));
    connect(exportAction, &QAction::triggered, this, &SnapshotBrowserWidget::onExportSnapshot);

    QAction* branchAction = menu.addAction(tr("Branch From Here..."));
    connect(branchAction, &QAction::triggered, this, &SnapshotBrowserWidget::onBranchFromSnapshot);

    menu.addSeparator();

    QAction* deleteAction = menu.addAction(tr("Delete Snapshot"));
    connect(deleteAction, &QAction::triggered, this, &SnapshotBrowserWidget::onDeleteSnapshot);

    menu.exec(listWidget_->viewport()->mapToGlobal(pos));
}

void SnapshotBrowserWidget::onExportSnapshot()
{
    int iteration = selectedIteration();
    if (iteration < 0) return;

    QString defaultName = QString("%1_iter%2.dat")
        .arg(currentRunId_)
        .arg(iteration, 6, 10, QChar('0'));

    QString path = QFileDialog::getSaveFileName(this,
        tr("Export Snapshot"),
        defaultName,
        tr("Data files (*.dat);;All files (*)"));

    if (path.isEmpty()) return;

    // TODO: Implement export using db_export_ascii
    QMessageBox::information(this, tr("Export"),
        tr("Export functionality not yet implemented."));
}

void SnapshotBrowserWidget::onDeleteSnapshot()
{
    int iteration = selectedIteration();
    if (iteration < 0) return;

    QMessageBox::information(this, tr("Delete"),
        tr("Individual snapshot deletion not yet implemented.\n"
           "Use session deletion to remove all snapshots."));
}

void SnapshotBrowserWidget::onBranchFromSnapshot()
{
    int iteration = selectedIteration();
    if (iteration < 0) return;

    // This would emit a signal to MainWindow to open the run dialog
    // with this snapshot pre-selected
    QMessageBox::information(this, tr("Branch"),
        tr("Use the Run Control panel to start a new simulation\n"
           "from this snapshot."));
}

} // namespace salr
