/*
 * HeadlessController.cpp - Headless CLI controller for SALR Visualization
 */

#include "HeadlessController.h"

#include <cstdio>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QTextStream>

namespace salr {

HeadlessController::HeadlessController(const Options& options, QObject* parent)
    : QObject(parent), options_(options) {}

void HeadlessController::start()
{
    QString dbPath = options_.databasePath.trimmed();
    if (dbPath.isEmpty()) {
        dbPath = QDir::current().filePath("database");
    }

    if (!database_.initialize(dbPath)) {
        logEvent("CLI_ERROR", {{"message", "Database initialization failed"}, {"database", dbPath}});
        QCoreApplication::quit();
        return;
    }

    runner_.setDatabasePath(dbPath);

    visualization_ = new VisualizationWidget();
    QSize renderSize = options_.renderSize.isValid() ? options_.renderSize : QSize(1200, 900);
    visualization_->setFixedSize(renderSize); // Force the size
    visualization_->setAttribute(Qt::WA_DontShowOnScreen); // Explicitly tell Qt not to show it
    visualization_->ensurePolished(); // Initialize style/fonts

    stdinFile_.open(stdin, QIODevice::ReadOnly | QIODevice::Text);
    stdinNotifier_ = new QSocketNotifier(fileno(stdin), QSocketNotifier::Read, this);
    connect(stdinNotifier_, &QSocketNotifier::activated, this, &HeadlessController::onStdinActivated);

    connect(&runner_, &SimulationRunner::started, this, &HeadlessController::onSimulationStarted);
    connect(&runner_, &SimulationRunner::progress, this, &HeadlessController::onSimulationProgress);
    connect(&runner_, &SimulationRunner::finished, this, &HeadlessController::onSimulationFinished);
    connect(&runner_, &SimulationRunner::errorOccurred, this, &HeadlessController::onSimulationError);

    snapshotTimer_.setInterval(1000);
    connect(&snapshotTimer_, &QTimer::timeout, this, &HeadlessController::onSnapshotPoll);

    if (!options_.sessionId.trimmed().isEmpty()) {
        openSession(options_.sessionId.trimmed());
    }

    if (!options_.configPath.trimmed().isEmpty()) {
        startSimulation();
    }

    logEvent("CLI_READY", {{"database", dbPath}});
}

void HeadlessController::onStdinActivated()
{
    while (stdinFile_.canReadLine()) {
        QString line = QString::fromUtf8(stdinFile_.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }

        QString command = line.section(' ', 0, 0).trimmed().toUpper();
        QString argLine = line.mid(command.length()).trimmed();

        if (command == "OPEN_SESSION") {
            bool ok = openSession(argLine);
            logEvent(ok ? "CLI_CMD_OK" : "CLI_CMD_ERROR", {{"command", command}, {"arg", argLine}});
        } else if (command == "LOAD_SNAPSHOT") {
            bool ok = loadSnapshot(argLine);
            logEvent(ok ? "CLI_CMD_OK" : "CLI_CMD_ERROR", {{"command", command}, {"arg", argLine}});
        } else if (command == "EXPORT_VISUALS") {
            bool ok = exportVisuals(argLine);
            logEvent(ok ? "CLI_CMD_OK" : "CLI_CMD_ERROR", {{"command", command}, {"arg", argLine}});
        } else if (command == "HELP") {
            logEvent("CLI_HELP", {{"commands", "OPEN_SESSION, LOAD_SNAPSHOT, EXPORT_VISUALS, QUIT"}});
        } else if (command == "QUIT") {
            logEvent("CLI_QUIT");
            QCoreApplication::quit();
            return;
        } else {
            logEvent("CLI_CMD_ERROR", {{"command", command}, {"message", "Unknown command"}});
        }
    }
}

void HeadlessController::onSimulationStarted(const QString& runId)
{
    if (runId.isEmpty() || runId == "starting...") {
        return;
    }

    currentRunId_ = runId;
    knownSnapshots_.clear();
    for (int iter : database_.listSnapshots(runId)) {
        knownSnapshots_.insert(iter);
    }

    snapshotTimer_.start();
    logEvent("CLI_SESSION_STARTED", {{"run_id", runId}, {"backend", backendLabel_}});
}

void HeadlessController::onSimulationProgress(int iteration, double error, double deltaError)
{
    Q_UNUSED(iteration);
    Q_UNUSED(error);
    Q_UNUSED(deltaError);
}

void HeadlessController::onSimulationFinished(bool converged, const QString& runId)
{
    snapshotTimer_.stop();
    logEvent("CLI_SESSION_FINISHED", {{"run_id", runId}, {"converged", converged ? "true" : "false"}});
}

void HeadlessController::onSimulationError(const QString& message)
{
    logEvent("CLI_ERROR", {{"message", message}});
}

void HeadlessController::onSnapshotPoll()
{
    if (currentRunId_.isEmpty()) {
        return;
    }

    QList<int> snapshots = database_.listSnapshots(currentRunId_);
    for (int iter : snapshots) {
        if (!knownSnapshots_.contains(iter)) {
            knownSnapshots_.insert(iter);
            logEvent("CLI_SNAPSHOT_WRITTEN", {{"run_id", currentRunId_}, {"iteration", QString::number(iter)}});
        }
    }
}

void HeadlessController::logEvent(const QString& event,
                                  const std::initializer_list<QPair<QString, QString>>& fields)
{
    QMap<QString, QString> mapFields;
    for (const auto& field : fields) {
        mapFields.insert(field.first, field.second);
    }

    QTextStream out(stdout);
    out << event;
    for (auto it = mapFields.constBegin(); it != mapFields.constEnd(); ++it) {
        out << ' ' << it.key() << '=' << it.value();
    }
    out << '\n';
    out.flush();
}

void HeadlessController::logConfig(const SimulationConfig& config, const QString& backend, const QString& configPath)
{
    logEvent("CLI_INIT", {
        {"backend", backend},
        {"config_path", configPath},
        {"grid_nx", QString::number(config.grid.nx)},
        {"grid_ny", QString::number(config.grid.ny)},
        {"grid_dx", QString::number(config.grid.dx)},
        {"grid_dy", QString::number(config.grid.dy)},
        {"boundary_mode", boundaryModeToString(config.boundaryMode)},
        {"init_mode", config.initMode},
        {"temperature", QString::number(config.temperature)},
        {"rho1", QString::number(config.rho1)},
        {"rho2", QString::number(config.rho2)},
        {"cutoff_radius", QString::number(config.potential.cutoffRadius)},
        {"max_iterations", QString::number(config.solver.maxIterations)},
        {"tolerance", QString::number(config.solver.tolerance)},
        {"xi1", QString::number(config.solver.xi1)},
        {"xi2", QString::number(config.solver.xi2)},
        {"error_change_threshold", QString::number(config.solver.errorChangeThreshold)},
        {"xi_damping_factor", QString::number(config.solver.xiDampingFactor)},
        {"save_every", QString::number(config.saveEvery)},
        {"output_dir", config.outputDir}
    });
}

void HeadlessController::startSimulation()
{
    SimulationConfig config;
    QString configPath = options_.configPath.trimmed();
    if (!database_.loadConfigFile(configPath, config)) {
        logEvent("CLI_ERROR", {{"message", "Failed to load config"}, {"config_path", configPath}});
        return;
    }

    QString backend = options_.backend.trimmed().toLower();
    bool useCuda = false;
    if (backend.isEmpty() || backend == "cpu") {
        backendLabel_ = "cpu";
    } else if (backend == "cuda" || backend == "gpu") {
        backendLabel_ = "cuda";
        useCuda = true;
    } else {
        logEvent("CLI_ERROR", {{"message", "Unknown backend"}, {"backend", backend}});
        return;
    }

    currentConfig_ = config;
    configLoaded_ = true;

    logConfig(config, backendLabel_, configPath);

    runner_.setExecutablePath(database_.executablePath(useCuda));
    runner_.startNew(config, useCuda);
}

bool HeadlessController::openSession(const QString& runId)
{
    QString trimmed = runId.trimmed();
    if (trimmed.isEmpty()) {
        return false;
    }

    if (!database_.sessionExists(trimmed)) {
        logEvent("CLI_ERROR", {{"message", "Session not found"}, {"run_id", trimmed}});
        return false;
    }

    currentRunId_ = trimmed;
    knownSnapshots_.clear();
    for (int iter : database_.listSnapshots(trimmed)) {
        knownSnapshots_.insert(iter);
    }

    logEvent("CLI_SESSION_OPENED", {{"run_id", trimmed}, {"snapshot_count", QString::number(knownSnapshots_.size())}});
    return true;
}

bool HeadlessController::loadSnapshot(const QString& iterationToken)
{
    if (currentRunId_.isEmpty()) {
        logEvent("CLI_ERROR", {{"message", "No session open"}});
        return false;
    }

    QString token = iterationToken.trimmed();
    int iteration = -1;
    if (token.isEmpty() || token.toLower() == "latest") {
        iteration = -1;
    } else {
        bool ok = false;
        iteration = token.toInt(&ok);
        if (!ok) {
            logEvent("CLI_ERROR", {{"message", "Invalid iteration"}, {"iteration", token}});
            return false;
        }
    }

    SnapshotData data = database_.loadSnapshot(currentRunId_, iteration);
    if (!data.isValid()) {
        logEvent("CLI_ERROR", {{"message", "Failed to load snapshot"}, {"run_id", currentRunId_}});
        return false;
    }

    visualization_->setSnapshotData(data);
    currentIteration_ = data.meta.iteration;

    logEvent("CLI_SNAPSHOT_LOADED", {{"run_id", currentRunId_}, {"iteration", QString::number(currentIteration_)}});
    QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
    return true;
}

bool HeadlessController::exportVisuals(const QString& path)
{
    if (!visualization_) {
        return false;
    }

    QString scatterPath;
    QString heatmapPath;
    bool ok = visualization_->exportVisuals(path, &scatterPath, &heatmapPath);
    if (ok) {
        logEvent("CLI_EXPORT_DONE", {{"scatter", scatterPath}, {"heatmap", heatmapPath}});
    }
    return ok;
}

} // namespace salr
