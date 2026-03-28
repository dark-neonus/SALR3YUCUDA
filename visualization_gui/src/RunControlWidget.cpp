/*
 * RunControlWidget.cpp - Simulation launch control panel implementation
 */

#include "RunControlWidget.h"
#include "DatabaseWrapper.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QCoreApplication>

namespace salr {

RunControlWidget::RunControlWidget(DatabaseWrapper* database, QWidget* parent)
    : QWidget(parent)
    , database_(database)
{
    setupUi();
    onLoadDefault();  // Initialize with default values
}

void RunControlWidget::setupUi()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(4, 4, 4, 4);

    // Grid parameters group
    QGroupBox* gridGroup = new QGroupBox(tr("Grid"));
    QFormLayout* gridLayout = new QFormLayout(gridGroup);

    nxSpin_ = new QSpinBox();
    nxSpin_->setRange(8, 1024);
    nxSpin_->setValue(160);
    gridLayout->addRow("nx:", nxSpin_);

    nySpin_ = new QSpinBox();
    nySpin_->setRange(8, 1024);
    nySpin_->setValue(160);
    gridLayout->addRow("ny:", nySpin_);

    dxSpin_ = new QDoubleSpinBox();
    dxSpin_->setRange(0.01, 10.0);
    dxSpin_->setDecimals(3);
    dxSpin_->setSingleStep(0.05);
    dxSpin_->setValue(0.2);
    gridLayout->addRow("dx:", dxSpin_);

    dySpin_ = new QDoubleSpinBox();
    dySpin_->setRange(0.01, 10.0);
    dySpin_->setDecimals(3);
    dySpin_->setSingleStep(0.05);
    dySpin_->setValue(0.2);
    gridLayout->addRow("dy:", dySpin_);

    boundaryCombo_ = new QComboBox();
    boundaryCombo_->addItem("PBC", "PBC");
    boundaryCombo_->addItem("W2 (walls X)", "W2");
    boundaryCombo_->addItem("W4 (walls all)", "W4");
    gridLayout->addRow(tr("Boundary:"), boundaryCombo_);

    mainLayout->addWidget(gridGroup);

    // Physics parameters group
    QGroupBox* physicsGroup = new QGroupBox(tr("Physics"));
    QFormLayout* physicsLayout = new QFormLayout(physicsGroup);

    tempSpin_ = new QDoubleSpinBox();
    tempSpin_->setRange(0.1, 100.0);
    tempSpin_->setDecimals(2);
    tempSpin_->setValue(2.9);
    physicsLayout->addRow("T:", tempSpin_);

    rho1Spin_ = new QDoubleSpinBox();
    rho1Spin_->setRange(0.01, 10.0);
    rho1Spin_->setDecimals(3);
    rho1Spin_->setValue(0.4);
    physicsLayout->addRow("rho1:", rho1Spin_);

    rho2Spin_ = new QDoubleSpinBox();
    rho2Spin_->setRange(0.01, 10.0);
    rho2Spin_->setDecimals(3);
    rho2Spin_->setValue(0.2);
    physicsLayout->addRow("rho2:", rho2Spin_);

    cutoffSpin_ = new QDoubleSpinBox();
    cutoffSpin_->setRange(1.0, 100.0);
    cutoffSpin_->setDecimals(1);
    cutoffSpin_->setValue(16.0);
    physicsLayout->addRow(tr("Cutoff:"), cutoffSpin_);

    mainLayout->addWidget(physicsGroup);

    // Solver parameters group
    QGroupBox* solverGroup = new QGroupBox(tr("Solver"));
    QFormLayout* solverLayout = new QFormLayout(solverGroup);

    maxIterSpin_ = new QSpinBox();
    maxIterSpin_->setRange(100, 1000000);
    maxIterSpin_->setSingleStep(1000);
    maxIterSpin_->setValue(50000);
    solverLayout->addRow(tr("Max iter:"), maxIterSpin_);

    tolSpin_ = new QDoubleSpinBox();
    tolSpin_->setRange(1e-12, 1e-2);
    tolSpin_->setDecimals(10);
    tolSpin_->setValue(1e-8);
    tolSpin_->setSpecialValueText("1e-8");
    solverLayout->addRow(tr("Tolerance:"), tolSpin_);

    xi1Spin_ = new QDoubleSpinBox();
    xi1Spin_->setRange(0.001, 1.0);
    xi1Spin_->setDecimals(4);
    xi1Spin_->setSingleStep(0.01);
    xi1Spin_->setValue(0.2);
    solverLayout->addRow("xi1:", xi1Spin_);

    xi2Spin_ = new QDoubleSpinBox();
    xi2Spin_->setRange(0.001, 1.0);
    xi2Spin_->setDecimals(4);
    xi2Spin_->setSingleStep(0.01);
    xi2Spin_->setValue(0.2);
    solverLayout->addRow("xi2:", xi2Spin_);

    saveEverySpin_ = new QSpinBox();
    saveEverySpin_->setRange(10, 100000);
    saveEverySpin_->setSingleStep(100);
    saveEverySpin_->setValue(1000);
    solverLayout->addRow(tr("Save every:"), saveEverySpin_);

    mainLayout->addWidget(solverGroup);

    // Resume options
    QGroupBox* resumeGroup = new QGroupBox(tr("Resume/Branch"));
    QVBoxLayout* resumeLayout = new QVBoxLayout(resumeGroup);

    resumeCheck_ = new QCheckBox(tr("Start from snapshot"));
    resumeLayout->addWidget(resumeCheck_);

    QHBoxLayout* snapLayout = new QHBoxLayout();
    snapLayout->addWidget(new QLabel(tr("Snapshot:")));
    snapshotCombo_ = new QComboBox();
    snapshotCombo_->setEnabled(false);
    snapLayout->addWidget(snapshotCombo_, 1);
    resumeLayout->addLayout(snapLayout);

    connect(resumeCheck_, &QCheckBox::toggled, snapshotCombo_, &QComboBox::setEnabled);

    mainLayout->addWidget(resumeGroup);

    // Load buttons
    QHBoxLayout* loadLayout = new QHBoxLayout();

    loadDefaultBtn_ = new QPushButton(tr("Load Default"));
    connect(loadDefaultBtn_, &QPushButton::clicked, this, &RunControlWidget::onLoadDefault);
    loadLayout->addWidget(loadDefaultBtn_);

    loadSessionBtn_ = new QPushButton(tr("Load from Session"));
    connect(loadSessionBtn_, &QPushButton::clicked, this, &RunControlWidget::onLoadFromSession);
    loadLayout->addWidget(loadSessionBtn_);

    mainLayout->addLayout(loadLayout);

    // Action buttons
    QHBoxLayout* actionLayout = new QHBoxLayout();

    startCpuBtn_ = new QPushButton(tr("Start CPU"));
    startCpuBtn_->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }");
    connect(startCpuBtn_, &QPushButton::clicked, this, &RunControlWidget::onStartCpu);
    actionLayout->addWidget(startCpuBtn_);

    startCudaBtn_ = new QPushButton(tr("Start CUDA"));
    startCudaBtn_->setStyleSheet("QPushButton { background-color: #2196F3; color: white; }");
    connect(startCudaBtn_, &QPushButton::clicked, this, &RunControlWidget::onStartCuda);
    actionLayout->addWidget(startCudaBtn_);

    mainLayout->addLayout(actionLayout);

    stopBtn_ = new QPushButton(tr("Stop Simulation"));
    stopBtn_->setStyleSheet("QPushButton { background-color: #f44336; color: white; }");
    stopBtn_->setEnabled(false);
    connect(stopBtn_, &QPushButton::clicked, this, &RunControlWidget::onStop);
    mainLayout->addWidget(stopBtn_);

    mainLayout->addStretch();
}

void RunControlWidget::setCurrentSession(const QString& runId)
{
    currentRunId_ = runId;

    // Populate snapshot combo for resume
    snapshotCombo_->clear();

    if (!runId.isEmpty() && database_->isInitialized()) {
        QVector<int> snapshots = database_->listSnapshots(runId);
        for (int iter : snapshots) {
            snapshotCombo_->addItem(QString("Iteration %1").arg(iter), iter);
        }

        // Select latest by default
        if (snapshotCombo_->count() > 0) {
            snapshotCombo_->setCurrentIndex(snapshotCombo_->count() - 1);
        }
    }

    loadSessionBtn_->setEnabled(!runId.isEmpty());
}

void RunControlWidget::setRunning(bool running)
{
    isRunning_ = running;

    startCpuBtn_->setEnabled(!running);
    startCudaBtn_->setEnabled(!running);
    stopBtn_->setEnabled(running);

    // Disable config editing while running
    nxSpin_->setEnabled(!running);
    nySpin_->setEnabled(!running);
    dxSpin_->setEnabled(!running);
    dySpin_->setEnabled(!running);
    boundaryCombo_->setEnabled(!running);
    tempSpin_->setEnabled(!running);
    rho1Spin_->setEnabled(!running);
    rho2Spin_->setEnabled(!running);
    cutoffSpin_->setEnabled(!running);
    maxIterSpin_->setEnabled(!running);
    tolSpin_->setEnabled(!running);
    xi1Spin_->setEnabled(!running);
    xi2Spin_->setEnabled(!running);
    saveEverySpin_->setEnabled(!running);
    resumeCheck_->setEnabled(!running);
    loadDefaultBtn_->setEnabled(!running);
    loadSessionBtn_->setEnabled(!running && !currentRunId_.isEmpty());
}

void RunControlWidget::onLoadDefault()
{
    // Try to find default.cfg
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList searchPaths = {
        appDir + "/configs/default.cfg",
        appDir + "/../configs/default.cfg",
        appDir + "/../../configs/default.cfg",
        QDir::currentPath() + "/configs/default.cfg"
    };

    SimulationConfig config;
    bool loaded = false;

    for (const QString& path : searchPaths) {
        if (QFile::exists(path) && database_->loadConfigFile(path, config)) {
            loaded = true;
            break;
        }
    }

    if (loaded) {
        applyConfig(config);
    } else {
        // Use hardcoded defaults
        config = SimulationConfig();
        applyConfig(config);
    }
}

void RunControlWidget::onLoadFromSession()
{
    if (currentRunId_.isEmpty()) {
        return;
    }

    SessionInfo info = database_->getSessionInfo(currentRunId_);
    SimulationConfig config = database_->configFromSession(info);
    applyConfig(config);
}

void RunControlWidget::onStartCpu()
{
    SimulationConfig config = buildConfig();

    if (resumeCheck_->isChecked() && snapshotCombo_->currentIndex() >= 0) {
        int iteration = snapshotCombo_->currentData().toInt();
        emit resumeSimulation(currentRunId_, iteration, false);
    } else {
        emit startSimulation(config, false);
    }
}

void RunControlWidget::onStartCuda()
{
    SimulationConfig config = buildConfig();

    if (resumeCheck_->isChecked() && snapshotCombo_->currentIndex() >= 0) {
        int iteration = snapshotCombo_->currentData().toInt();
        emit resumeSimulation(currentRunId_, iteration, true);
    } else {
        emit startSimulation(config, true);
    }
}

void RunControlWidget::onResume()
{
    if (currentRunId_.isEmpty() || snapshotCombo_->currentIndex() < 0) {
        return;
    }

    int iteration = snapshotCombo_->currentData().toInt();
    emit resumeSimulation(currentRunId_, iteration, false);
}

void RunControlWidget::onStop()
{
    emit stopSimulation();
}

SimulationConfig RunControlWidget::buildConfig() const
{
    SimulationConfig config;

    config.grid.nx = nxSpin_->value();
    config.grid.ny = nySpin_->value();
    config.grid.dx = dxSpin_->value();
    config.grid.dy = dySpin_->value();
    config.grid.computeDerivedValues();

    config.boundaryMode = stringToBoundaryMode(boundaryCombo_->currentData().toString());

    config.temperature = tempSpin_->value();
    config.rho1 = rho1Spin_->value();
    config.rho2 = rho2Spin_->value();
    config.potential.cutoffRadius = cutoffSpin_->value();

    config.solver.maxIterations = maxIterSpin_->value();
    config.solver.tolerance = tolSpin_->value();
    config.solver.xi1 = xi1Spin_->value();
    config.solver.xi2 = xi2Spin_->value();

    config.saveEvery = saveEverySpin_->value();

    return config;
}

void RunControlWidget::applyConfig(const SimulationConfig& config)
{
    nxSpin_->setValue(config.grid.nx);
    nySpin_->setValue(config.grid.ny);
    dxSpin_->setValue(config.grid.dx);
    dySpin_->setValue(config.grid.dy);

    int bcIndex = boundaryCombo_->findData(boundaryModeToString(config.boundaryMode));
    if (bcIndex >= 0) {
        boundaryCombo_->setCurrentIndex(bcIndex);
    }

    tempSpin_->setValue(config.temperature);
    rho1Spin_->setValue(config.rho1);
    rho2Spin_->setValue(config.rho2);
    cutoffSpin_->setValue(config.potential.cutoffRadius);

    maxIterSpin_->setValue(config.solver.maxIterations);
    tolSpin_->setValue(config.solver.tolerance);
    xi1Spin_->setValue(config.solver.xi1);
    xi2Spin_->setValue(config.solver.xi2);

    saveEverySpin_->setValue(config.saveEvery);
}

} // namespace salr
