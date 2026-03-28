/*
 * ConfigEditorDialog.cpp - Full configuration editor dialog implementation
 */

#include "ConfigEditorDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QMessageBox>

namespace salr {

ConfigEditorDialog::ConfigEditorDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Configuration Editor"));
    setMinimumSize(600, 500);
    setupUi();
}

void ConfigEditorDialog::setupUi()
{
    QVBoxLayout* layout = new QVBoxLayout(this);

    // Preview text
    previewEdit_ = new QTextEdit();
    previewEdit_->setReadOnly(true);
    previewEdit_->setFontFamily("monospace");
    layout->addWidget(previewEdit_, 1);

    // Load/Save buttons
    QHBoxLayout* fileLayout = new QHBoxLayout();

    QPushButton* loadBtn = new QPushButton(tr("Load from File..."));
    connect(loadBtn, &QPushButton::clicked, this, &ConfigEditorDialog::onLoadFile);
    fileLayout->addWidget(loadBtn);

    QPushButton* saveBtn = new QPushButton(tr("Save to File..."));
    connect(saveBtn, &QPushButton::clicked, this, &ConfigEditorDialog::onSaveFile);
    fileLayout->addWidget(saveBtn);

    fileLayout->addStretch();
    layout->addLayout(fileLayout);

    // Dialog buttons
    QDialogButtonBox* buttonBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttonBox);

    updatePreview();
}

void ConfigEditorDialog::setConfig(const SimulationConfig& config)
{
    config_ = config;
    updatePreview();
}

SimulationConfig ConfigEditorDialog::config() const
{
    return config_;
}

void ConfigEditorDialog::onLoadFile()
{
    QString path = QFileDialog::getOpenFileName(this,
        tr("Load Configuration"),
        QString(),
        tr("Config files (*.cfg);;All files (*)"));

    if (path.isEmpty()) return;

    // TODO: Parse config file
    QMessageBox::information(this, tr("Load"), tr("Config loading not yet implemented"));
}

void ConfigEditorDialog::onSaveFile()
{
    QString path = QFileDialog::getSaveFileName(this,
        tr("Save Configuration"),
        "config.cfg",
        tr("Config files (*.cfg);;All files (*)"));

    if (path.isEmpty()) return;

    // TODO: Save config file
    QMessageBox::information(this, tr("Save"), tr("Config saving not yet implemented"));
}

void ConfigEditorDialog::updatePreview()
{
    QString preview;
    QTextStream out(&preview);

    out << "[grid]\n";
    out << "nx = " << config_.grid.nx << "\n";
    out << "ny = " << config_.grid.ny << "\n";
    out << "dx = " << config_.grid.dx << "\n";
    out << "dy = " << config_.grid.dy << "\n";
    out << "boundary_mode = " << boundaryModeToString(config_.boundaryMode) << "\n";
    out << "\n";

    out << "[physics]\n";
    out << "temperature = " << config_.temperature << "\n";
    out << "rho1 = " << config_.rho1 << "\n";
    out << "rho2 = " << config_.rho2 << "\n";
    out << "cutoff_radius = " << config_.potential.cutoffRadius << "\n";
    out << "\n";

    out << "[solver]\n";
    out << "max_iterations = " << config_.solver.maxIterations << "\n";
    out << "tolerance = " << config_.solver.tolerance << "\n";
    out << "xi1 = " << config_.solver.xi1 << "\n";
    out << "xi2 = " << config_.solver.xi2 << "\n";
    out << "\n";

    out << "[output]\n";
    out << "save_every = " << config_.saveEvery << "\n";

    previewEdit_->setText(preview);
}

} // namespace salr
