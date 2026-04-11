/*
 * MainWindow.cpp - Main application window implementation
 */

#include "MainWindow.h"
#include "DatabaseWrapper.h"
#include "SessionBrowserWidget.h"
#include "SnapshotBrowserWidget.h"
#include "VisualizationWidget.h"
#include "ParameterDisplayWidget.h"
#include "RunControlWidget.h"
#include "SimulationRunner.h"

#include <QMenuBar>
#include <QToolBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QCloseEvent>
#include <QVBoxLayout>
#include <QApplication>

namespace salr {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle(tr("SALR DFT Visualization"));
    setMinimumSize(1200, 800);

    // Create core components
    database_ = std::make_unique<DatabaseWrapper>(this);
    simulationRunner_ = std::make_unique<SimulationRunner>(this);

    setupUi();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();
    setupConnections();

    loadSettings();

    // Try to initialize with default database path
    if (databasePath_.isEmpty()) {
        databasePath_ = QDir::currentPath() + "/database";
    }

    if (QDir(databasePath_).exists()) {
        if (database_->initialize(databasePath_)) {
            refreshSessionList();
            showMessage(tr("Database loaded: %1").arg(databasePath_));
        }
    }
}

MainWindow::~MainWindow()
{
    saveSettings();
}

void MainWindow::setupUi()
{
    // Create main splitter (horizontal)
    mainSplitter_ = new QSplitter(Qt::Horizontal, this);
    setCentralWidget(mainSplitter_);

    // Left panel: session and snapshot browsers (vertical splitter)
    leftSplitter_ = new QSplitter(Qt::Vertical);
    leftSplitter_->setMinimumWidth(280);
    leftSplitter_->setMaximumWidth(400);

    sessionBrowser_ = new SessionBrowserWidget(database_.get(), this);
    snapshotBrowser_ = new SnapshotBrowserWidget(database_.get(), this);

    leftSplitter_->addWidget(sessionBrowser_);
    leftSplitter_->addWidget(snapshotBrowser_);
    leftSplitter_->setStretchFactor(0, 2);
    leftSplitter_->setStretchFactor(1, 1);

    mainSplitter_->addWidget(leftSplitter_);

    // Center: visualization widget
    visualization_ = new VisualizationWidget(this);
    mainSplitter_->addWidget(visualization_);

    // Right dock: parameters and run control
    rightDock_ = new QDockWidget(tr("Controls"), this);
    rightDock_->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    rightDock_->setMinimumWidth(300);

    QWidget* rightPanel = new QWidget();
    QVBoxLayout* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(0, 0, 0, 0);

    parameterDisplay_ = new ParameterDisplayWidget(this);
    runControl_ = new RunControlWidget(database_.get(), this);

    rightLayout->addWidget(parameterDisplay_, 1);
    rightLayout->addWidget(runControl_, 0);

    rightDock_->setWidget(rightPanel);
    addDockWidget(Qt::RightDockWidgetArea, rightDock_);

    // Set splitter proportions
    mainSplitter_->setStretchFactor(0, 0);  // Left panel: fixed
    mainSplitter_->setStretchFactor(1, 1);  // Visualization: stretch
}

void MainWindow::setupMenuBar()
{
    QMenuBar* menuBar = this->menuBar();

    // File menu
    QMenu* fileMenu = menuBar->addMenu(tr("&File"));

    QAction* openDbAction = fileMenu->addAction(tr("&Open Database..."));
    openDbAction->setShortcut(QKeySequence::Open);
    connect(openDbAction, &QAction::triggered, this, &MainWindow::onOpenDatabase);

    fileMenu->addSeparator();

    QAction* exportAction = fileMenu->addAction(tr("&Export Snapshot..."));
    exportAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_E));
    connect(exportAction, &QAction::triggered, this, &MainWindow::onExportSnapshot);

    fileMenu->addSeparator();

    QAction* quitAction = fileMenu->addAction(tr("&Quit"));
    quitAction->setShortcut(QKeySequence::Quit);
    connect(quitAction, &QAction::triggered, this, &MainWindow::onQuit);

    // View menu
    QMenu* viewMenu = menuBar->addMenu(tr("&View"));

    QAction* refreshAction = viewMenu->addAction(tr("&Refresh"));
    refreshAction->setShortcut(QKeySequence::Refresh);
    connect(refreshAction, &QAction::triggered, this, &MainWindow::refreshSessionList);

    viewMenu->addSeparator();

    QAction* toggleRightDock = rightDock_->toggleViewAction();
    toggleRightDock->setText(tr("Show &Controls Panel"));
    viewMenu->addAction(toggleRightDock);

    QAction* toggleParamsAction = viewMenu->addAction(tr("Show &Parameters Table"));
    toggleParamsAction->setCheckable(true);
    toggleParamsAction->setChecked(true);
    connect(toggleParamsAction, &QAction::toggled, parameterDisplay_, &QWidget::setVisible);

    // Simulation menu
    QMenu* simMenu = menuBar->addMenu(tr("&Simulation"));

    QAction* stopAction = simMenu->addAction(tr("&Stop Simulation"));
    stopAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Period));
    connect(stopAction, &QAction::triggered, this, &MainWindow::onStopSimulation);

    // Help menu
    QMenu* helpMenu = menuBar->addMenu(tr("&Help"));

    QAction* aboutAction = helpMenu->addAction(tr("&About"));
    connect(aboutAction, &QAction::triggered, this, [this]() {
        QMessageBox::about(this, tr("About SALR Visualization"),
            tr("SALR DFT Visualization GUI\n\n"
               "A Qt-based interface for visualizing and controlling\n"
               "SALR density functional theory simulations.\n\n"
               "Part of the SALR3YUCUDA project."));
    });
}

void MainWindow::setupToolBar()
{
    QToolBar* toolBar = addToolBar(tr("Main Toolbar"));
    toolBar->setMovable(false);

    QAction* refreshAction = toolBar->addAction(tr("Refresh"));
    refreshAction->setToolTip(tr("Refresh session list"));
    connect(refreshAction, &QAction::triggered, this, &MainWindow::refreshSessionList);

    toolBar->addSeparator();

    QAction* stopAction = toolBar->addAction(tr("Stop"));
    stopAction->setToolTip(tr("Stop running simulation"));
    connect(stopAction, &QAction::triggered, this, &MainWindow::onStopSimulation);
}

void MainWindow::setupStatusBar()
{
    QStatusBar* status = statusBar();

    statusLabel_ = new QLabel(tr("Ready"));
    status->addWidget(statusLabel_, 1);

    progressBar_ = new QProgressBar();
    progressBar_->setMaximumWidth(200);
    progressBar_->setVisible(false);
    status->addWidget(progressBar_);

    sessionLabel_ = new QLabel();
    status->addPermanentWidget(sessionLabel_);
}

void MainWindow::setupConnections()
{
    // Database signals
    connect(database_.get(), &DatabaseWrapper::sessionListChanged,
            this, &MainWindow::refreshSessionList);
    connect(database_.get(), &DatabaseWrapper::errorOccurred,
            this, &MainWindow::showError);

    // Session browser signals
    connect(sessionBrowser_, &SessionBrowserWidget::sessionSelected,
            this, &MainWindow::onSessionSelected);
    connect(sessionBrowser_, &SessionBrowserWidget::sessionDoubleClicked,
            this, &MainWindow::onSessionDoubleClicked);

    // Snapshot browser signals
    connect(snapshotBrowser_, &SnapshotBrowserWidget::snapshotSelected,
            this, &MainWindow::onSnapshotSelected);
    connect(snapshotBrowser_, &SnapshotBrowserWidget::snapshotDoubleClicked,
            this, &MainWindow::onSnapshotDoubleClicked);

    // Run control signals
    connect(runControl_, &RunControlWidget::startSimulation,
            this, &MainWindow::onStartSimulation);
    connect(runControl_, &RunControlWidget::resumeSimulation,
            this, &MainWindow::onResumeSimulation);
    connect(runControl_, &RunControlWidget::stopSimulation,
            this, &MainWindow::onStopSimulation);

    // Simulation runner signals
    connect(simulationRunner_.get(), &SimulationRunner::started,
            this, &MainWindow::onSimulationStarted);
    connect(simulationRunner_.get(), &SimulationRunner::progress,
            this, &MainWindow::onSimulationProgress);
    connect(simulationRunner_.get(), &SimulationRunner::finished,
            this, &MainWindow::onSimulationFinished);
    connect(simulationRunner_.get(), &SimulationRunner::errorOccurred,
            this, &MainWindow::onSimulationError);
}

void MainWindow::loadSettings()
{
    QSettings settings("SALR", "VisualizationGUI");

    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("windowState").toByteArray());

    databasePath_ = settings.value("databasePath").toString();

    if (mainSplitter_) {
        mainSplitter_->restoreState(settings.value("mainSplitter").toByteArray());
    }
    if (leftSplitter_) {
        leftSplitter_->restoreState(settings.value("leftSplitter").toByteArray());
    }
}

void MainWindow::saveSettings()
{
    QSettings settings("SALR", "VisualizationGUI");

    settings.setValue("geometry", saveGeometry());
    settings.setValue("windowState", saveState());
    settings.setValue("databasePath", databasePath_);

    if (mainSplitter_) {
        settings.setValue("mainSplitter", mainSplitter_->saveState());
    }
    if (leftSplitter_) {
        settings.setValue("leftSplitter", leftSplitter_->saveState());
    }
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (simulationRunner_->isRunning()) {
        int ret = QMessageBox::question(this, tr("Simulation Running"),
            tr("A simulation is currently running. Stop it and quit?"),
            QMessageBox::Yes | QMessageBox::No);

        if (ret == QMessageBox::No) {
            event->ignore();
            return;
        }

        simulationRunner_->stop();
    }

    saveSettings();
    event->accept();
}

void MainWindow::onOpenDatabase()
{
    QString dir = QFileDialog::getExistingDirectory(this,
        tr("Select Database Directory"),
        databasePath_.isEmpty() ? QDir::currentPath() : databasePath_);

    if (dir.isEmpty()) {
        return;
    }

    databasePath_ = dir;

    if (database_->initialize(databasePath_)) {
        refreshSessionList();
        showMessage(tr("Database loaded: %1").arg(databasePath_));
    } else {
        showError(tr("Failed to open database at %1").arg(databasePath_));
    }
}

void MainWindow::onExportSnapshot()
{
    if (currentRunId_.isEmpty() || currentIteration_ < 0) {
        showError(tr("No snapshot selected"));
        return;
    }

    QString defaultName = QString("%1_iter%2.dat")
        .arg(currentRunId_)
        .arg(currentIteration_, 6, 10, QChar('0'));

    QString path = QFileDialog::getSaveFileName(this,
        tr("Export Snapshot"),
        defaultName,
        tr("Data files (*.dat);;All files (*)"));

    if (path.isEmpty()) {
        return;
    }

    // TODO: Implement ASCII export
    showMessage(tr("Export not yet implemented"));
}

void MainWindow::onQuit()
{
    close();
}

void MainWindow::onSessionSelected(const QString& runId)
{
    currentRunId_ = runId;
    currentIteration_ = -1;

    sessionLabel_->setText(runId);

    // Load snapshots for this session
    snapshotBrowser_->setSession(runId);

    // Update parameter display with session info
    SessionInfo info = database_->getSessionInfo(runId);
    parameterDisplay_->setSessionInfo(info);

    // Update run control with session config
    runControl_->setCurrentSession(runId);
}

void MainWindow::onSessionDoubleClicked(const QString& runId)
{
    onSessionSelected(runId);

    // Also load the latest snapshot
    QVector<int> snapshots = database_->listSnapshots(runId);
    if (!snapshots.isEmpty()) {
        loadSnapshot(runId, snapshots.last());
    }
}

void MainWindow::onSnapshotSelected(int iteration)
{
    currentIteration_ = iteration;

    // Update status
    sessionLabel_->setText(QString("%1 @ %2")
        .arg(currentRunId_)
        .arg(iteration));
}

void MainWindow::onSnapshotDoubleClicked(int iteration)
{
    loadSnapshot(currentRunId_, iteration);
}

void MainWindow::loadSnapshot(const QString& runId, int iteration)
{
    if (runId.isEmpty()) {
        return;
    }

    showMessage(tr("Loading snapshot %1...").arg(iteration));

    SnapshotData data = database_->loadSnapshot(runId, iteration);

    if (!data.isValid()) {
        showError(tr("Failed to load snapshot"));
        return;
    }

    currentRunId_ = runId;
    currentIteration_ = iteration;

    // Update visualization
    visualization_->setSnapshotData(data);

    // Update parameter display
    parameterDisplay_->setSnapshotMeta(data.meta);

    showMessage(tr("Loaded snapshot: iteration %1, error %2")
        .arg(data.meta.iteration)
        .arg(data.meta.currentError, 0, 'e', 2));
}

void MainWindow::onStartSimulation(const SimulationConfig& config, bool useCuda)
{
    if (!database_->isInitialized()) {
        showError(tr("Database not initialized"));
        return;
    }

    simulationRunner_->setDatabasePath(databasePath_);
    simulationRunner_->setExecutablePath(database_->executablePath(useCuda));
    simulationRunner_->startNew(config, useCuda);
}

void MainWindow::onResumeSimulation(const QString& runId, int iteration, const SimulationConfig& config, bool useCuda)
{
    if (!database_->isInitialized()) {
        showError(tr("Database not initialized"));
        return;
    }

    simulationRunner_->setDatabasePath(databasePath_);
    simulationRunner_->setExecutablePath(database_->executablePath(useCuda));
    simulationRunner_->resume(runId, iteration, config, useCuda);
}

void MainWindow::onStopSimulation()
{
    if (simulationRunner_->isRunning()) {
        simulationRunner_->stop();
        showMessage(tr("Simulation stopped"));
    }
}

void MainWindow::onSimulationStarted(const QString& runId)
{
    showMessage(tr("Simulation started: %1").arg(runId));

    progressBar_->setVisible(true);
    progressBar_->setRange(0, 0);  // Indeterminate

    runControl_->setRunning(true);
}

void MainWindow::onSimulationProgress(int iteration, double error, double deltaError)
{
    Q_UNUSED(deltaError)

    statusLabel_->setText(tr("Iteration %1: error = %2")
        .arg(iteration)
        .arg(error, 0, 'e', 2));

    // Refresh snapshot list periodically
    static int lastRefreshIter = 0;
    if (iteration - lastRefreshIter >= 1000) {
        refreshSnapshotList();
        lastRefreshIter = iteration;
    }
}

void MainWindow::onSimulationFinished(bool converged, const QString& runId)
{
    progressBar_->setVisible(false);
    runControl_->setRunning(false);

    QString status = converged ? tr("converged") : tr("stopped");
    showMessage(tr("Simulation %1: %2").arg(status).arg(runId));

    // Refresh lists
    refreshSessionList();
    refreshSnapshotList();
}

void MainWindow::onSimulationError(const QString& message)
{
    progressBar_->setVisible(false);
    runControl_->setRunning(false);
    showError(message);
}

void MainWindow::refreshSessionList()
{
    if (database_->isInitialized()) {
        sessionBrowser_->refresh();
    }
}

void MainWindow::refreshSnapshotList()
{
    if (!currentRunId_.isEmpty()) {
        snapshotBrowser_->refresh();
    }
}

void MainWindow::showMessage(const QString& message, int timeout)
{
    statusBar()->showMessage(message, timeout);
}

void MainWindow::showError(const QString& message)
{
    QMessageBox::warning(this, tr("Error"), message);
}

} // namespace salr
