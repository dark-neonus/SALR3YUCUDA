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
#include <QScrollArea>

namespace salr {

RunControlWidget::RunControlWidget(DatabaseWrapper* database, QWidget* parent)
    : QWidget(parent)
    , database_(database)
{
    setupUi();
    onLoadDefault();
}

void RunControlWidget::setupUi()
{
    QVBoxLayout* outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(0, 0, 0, 0);

    scrollArea_ = new QScrollArea();
    scrollArea_->setWidgetResizable(true);
    scrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    QWidget* container = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(container);
    mainLayout->setContentsMargins(4, 4, 4, 4);

    // Grid parameters group (collapsible)
    gridGroup_ = new QGroupBox(tr("▼ Grid"));
    gridGroup_->setCheckable(true);
    gridGroup_->setChecked(true);
    QFormLayout* gridLayout = new QFormLayout();

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

    gridGroup_->setLayout(gridLayout);
    mainLayout->addWidget(gridGroup_);

    connect(gridGroup_, &QGroupBox::toggled, [this](bool checked) {
        gridGroup_->setTitle(checked ? tr("▼ Grid") : tr("▶ Grid"));
        for (auto* child : gridGroup_->findChildren<QWidget*>()) {
            child->setVisible(checked);
        }
    });

    // Physics parameters group (collapsible)
    physicsGroup_ = new QGroupBox(tr("▼ Physics"));
    physicsGroup_->setCheckable(true);
    physicsGroup_->setChecked(true);
    QFormLayout* physicsLayout = new QFormLayout();

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

    physicsGroup_->setLayout(physicsLayout);
    mainLayout->addWidget(physicsGroup_);

    connect(physicsGroup_, &QGroupBox::toggled, [this](bool checked) {
        physicsGroup_->setTitle(checked ? tr("▼ Physics") : tr("▶ Physics"));
        for (auto* child : physicsGroup_->findChildren<QWidget*>()) {
            child->setVisible(checked);
        }
    });

    // Solver parameters group (collapsible)
    solverGroup_ = new QGroupBox(tr("▼ Solver"));
    solverGroup_->setCheckable(true);
    solverGroup_->setChecked(true);
    QFormLayout* solverLayout = new QFormLayout();

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

    solverGroup_->setLayout(solverLayout);
    mainLayout->addWidget(solverGroup_);

    connect(solverGroup_, &QGroupBox::toggled, [this](bool checked) {
        solverGroup_->setTitle(checked ? tr("▼ Solver") : tr("▶ Solver"));
        for (auto* child : solverGroup_->findChildren<QWidget*>()) {
            child->setVisible(checked);
        }
    });

    // Initial distribution group (collapsible)
    initGroup_ = new QGroupBox(tr("▼ Initial Distribution"));
    initGroup_->setCheckable(true);
    initGroup_->setChecked(true);
    QFormLayout* initLayout = new QFormLayout();

    initDistCombo_ = new QComboBox();
    initDistCombo_->addItem(tr("Random noise"), static_cast<int>(InitialDistribution::Random));
    initDistCombo_->addItem(tr("Sinusoidal pattern"), static_cast<int>(InitialDistribution::Sinusoids));
    initDistCombo_->addItem(tr("Uniform (trivial)"), static_cast<int>(InitialDistribution::Trivial));
    initLayout->addRow(tr("Type:"), initDistCombo_);

    initGroup_->setLayout(initLayout);
    mainLayout->addWidget(initGroup_);

    connect(initGroup_, &QGroupBox::toggled, [this](bool checked) {
        initGroup_->setTitle(checked ? tr("▼ Initial Distribution") : tr("▶ Initial Distribution"));
        for (auto* child : initGroup_->findChildren<QWidget*>()) {
            child->setVisible(checked);
        }
    });

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

    scrollArea_->setWidget(container);
    outerLayout->addWidget(scrollArea_);
}

InitialDistribution RunControlWidget::initialDistribution() const
{
    return static_cast<InitialDistribution>(initDistCombo_->currentData().toInt());
}

void RunControlWidget::setCurrentSession(const QString& runId)
{
    currentRunId_ = runId;

    snapshotCombo_->clear();

    if (!runId.isEmpty() && database_->isInitialized()) {
        QVector<int> snapshots = database_->listSnapshots(runId);
        for (int iter : snapshots) {
            snapshotCombo_->addItem(QString("Iteration %1").arg(iter), iter);
        }

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
    initDistCombo_->setEnabled(!running);
    resumeCheck_->setEnabled(!running);
    loadDefaultBtn_->setEnabled(!running);
    loadSessionBtn_->setEnabled(!running && !currentRunId_.isEmpty());
}

void RunControlWidget::onLoadDefault()
{
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
        emit resumeSimulation(currentRunId_, iteration, config, false);
    } else {
        emit startSimulation(config, false);
    }
}

void RunControlWidget::onStartCuda()
{
    SimulationConfig config = buildConfig();

    if (resumeCheck_->isChecked() && snapshotCombo_->currentIndex() >= 0) {
        int iteration = snapshotCombo_->currentData().toInt();
        emit resumeSimulation(currentRunId_, iteration, config, true);
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
    emit resumeSimulation(currentRunId_, iteration, buildConfig(), false);
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

    InitialDistribution initDist = static_cast<InitialDistribution>(initDistCombo_->currentData().toInt());
    if (initDist == InitialDistribution::Sinusoids)
        config.initMode = "sinusoids";
    else if (initDist == InitialDistribution::Trivial)
        config.initMode = "trivial";
    else
        config.initMode = "random";

    config.temperature = tempSpin_->value();
    config.rho1 = rho1Spin_->value();
    config.rho2 = rho2Spin_->value();

    config.potential = storedPotential_;
    config.potential.cutoffRadius = cutoffSpin_->value();

    // Preserve solver fields not exposed in the UI.
    config.solver = storedSolver_;
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

    storedPotential_ = config.potential;
    storedSolver_ = config.solver;

    maxIterSpin_->setValue(storedSolver_.maxIterations);
    tolSpin_->setValue(storedSolver_.tolerance);
    xi1Spin_->setValue(storedSolver_.xi1);
    xi2Spin_->setValue(storedSolver_.xi2);

    saveEverySpin_->setValue(config.saveEvery);
}

} // namespace salr
