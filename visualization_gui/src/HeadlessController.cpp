/*
 * HeadlessController.cpp - Headless CLI controller for SALR Visualization
 */

#include "HeadlessController.h"

#include <cstdio>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QTextStream>
#include <QSettings>
#include <iostream>

namespace salr {

HeadlessController::HeadlessController(const Options& options, QObject* parent)
    : QObject(parent), options_(options), currentIteration_(0) {}

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
    visualization_->setFixedSize(renderSize);
    visualization_->setAttribute(Qt::WA_DontShowOnScreen);
    visualization_->ensurePolished();

    stdinFile_.open(stdin, QIODevice::ReadOnly | QIODevice::Text);
    stdinNotifier_ = new QSocketNotifier(fileno(stdin), QSocketNotifier::Read, this);
    connect(stdinNotifier_, &QSocketNotifier::activated, this, &HeadlessController::onStdinActivated);

    connect(&runner_, &SimulationRunner::started, this, &HeadlessController::onSimulationStarted);
    connect(&runner_, &SimulationRunner::finished, this, &HeadlessController::onSimulationFinished);
    connect(&runner_, &SimulationRunner::errorOccurred, this, &HeadlessController::onSimulationError);

    logEvent("CLI_READY", {{"database", dbPath}});

    if (!options_.configPath.isEmpty()) {
        SimulationConfig config;
        
        QSettings settings(options_.configPath, QSettings::IniFormat);
        
        settings.beginGroup("grid");
        config.grid.nx = settings.value("nx", 80).toInt();
        config.grid.ny = settings.value("ny", 80).toInt();
        config.grid.dx = settings.value("dx", 0.2).toDouble();
        config.grid.dy = settings.value("dy", 0.2).toDouble();
        QString bc = settings.value("boundary_mode", "PBC").toString();
        if (bc == "W2") config.boundaryMode = BoundaryMode::W2;
        else if (bc == "W4") config.boundaryMode = BoundaryMode::W4;
        else config.boundaryMode = BoundaryMode::PBC;
        config.initMode = settings.value("init_mode", "random").toString();
        settings.endGroup();

        settings.beginGroup("physics");
        config.temperature = settings.value("temperature", 8.0).toDouble();
        config.rho1 = settings.value("rho1", 0.4).toDouble();
        config.rho2 = settings.value("rho2", 0.2).toDouble();
        config.potential.cutoffRadius = settings.value("cutoff_radius", 8.0).toDouble();
        settings.endGroup();

        settings.beginGroup("interaction");
        for(int i=0; i<3; ++i) {
            config.potential.A[0][0][i] = settings.value(QString("A_11_%1").arg(i+1), 0.0).toDouble();
            config.potential.alpha[0][0][i] = settings.value(QString("a_11_%1").arg(i+1), 1.0).toDouble();
            config.potential.A[0][1][i] = settings.value(QString("A_12_%1").arg(i+1), 0.0).toDouble();
            config.potential.alpha[0][1][i] = settings.value(QString("a_12_%1").arg(i+1), 1.0).toDouble();
            config.potential.A[1][1][i] = settings.value(QString("A_22_%1").arg(i+1), 0.0).toDouble();
            config.potential.alpha[1][1][i] = settings.value(QString("a_22_%1").arg(i+1), 1.0).toDouble();
        }
        settings.endGroup();

        settings.beginGroup("solver");
        config.solver.maxIterations = settings.value("max_iterations", 1000).toInt();
        config.solver.tolerance = settings.value("tolerance", 1e-6).toDouble();
        config.solver.xi1 = settings.value("xi1", 0.01).toDouble();
        config.solver.xi2 = settings.value("xi2", 0.01).toDouble();
        settings.endGroup();
        
        config.saveEvery = settings.value("output/save_every", 100).toInt();

        bool useCuda = (options_.backend.toLower() == "cuda");
        
        logEvent("CLI_INIT", {
            {"config_path", options_.configPath},
            {"backend", options_.backend}
        });

        runner_.startNew(config, useCuda);
    } else if (!options_.sessionId.isEmpty()) {
        currentRunId_ = options_.sessionId;
        loadSnapshot("latest");
    }
}

void HeadlessController::logEvent(const QString& eventName, const std::initializer_list<std::pair<QString, QString>>& params)
{
    std::cout << eventName.toStdString();
    for (const auto& pair : params) {
        std::cout << " " << pair.first.toStdString() << "=" << pair.second.toStdString();
    }
    std::cout << std::endl;
}

void HeadlessController::onStdinActivated()
{
    QByteArray lineData = stdinFile_.readLine();
    if (lineData.isEmpty()) {
        return;
    }

    QString line = QString::fromUtf8(lineData).trimmed();
    if (line.isEmpty()) return;

    QStringList parts = line.split(' ', Qt::SkipEmptyParts);
    QString cmd = parts[0].toUpper();

    if (cmd == "QUIT" || cmd == "EXIT") {
        logEvent("CLI_QUITTING", {});
        QCoreApplication::quit();
    } else if (cmd == "EXPORT_VISUALS") {
        if (parts.size() > 1) {
            exportVisuals(parts[1]);
        } else {
            logEvent("CLI_ERROR", {{"message", "EXPORT_VISUALS requires path argument"}});
        }
    } else if (cmd == "LOAD_SNAPSHOT") {
        if (parts.size() > 1) {
            loadSnapshot(parts[1]);
        }
    }
}

void HeadlessController::onSimulationStarted(const QString& runId)
{
    currentRunId_ = runId;
    logEvent("CLI_SESSION_STARTED", {{"run_id", currentRunId_}});
}

void HeadlessController::onSimulationFinished(bool converged, const QString& runId)
{
    QString finalRunId = runId.isEmpty() ? (currentRunId_.isEmpty() ? "none" : currentRunId_) : runId;
    
    logEvent("CLI_SESSION_FINISHED", {
        {"converged", converged ? "true" : "false"}, 
        {"run_id", finalRunId}
    });

    if (!finalRunId.isEmpty() && finalRunId != "none") {
        loadSnapshot("latest");
    }
}

void HeadlessController::onSimulationError(const QString& message)
{
    logEvent("CLI_ERROR", {{"message", message}});
    logEvent("CLI_SESSION_FINISHED", {{"converged", "false"}, {"run_id", currentRunId_}});
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
        logEvent("CLI_ERROR", {{"message", "Visualization widget not initialized"}});
        return false;
    }

    QString scatterPath;
    QString heatmapPath;
    bool ok = visualization_->exportVisuals(path, &scatterPath, &heatmapPath);
    if (ok) {
        logEvent("CLI_VISUALS_EXPORTED", {
            {"base_path", path},
            {"scatter", scatterPath},
            {"heatmap", heatmapPath}
        });
        return true;
    } else {
        logEvent("CLI_ERROR", {{"message", "Failed to export visuals to " + path}});
        return false;
    }
}

// Re-added to satisfy MOC/Linker
void HeadlessController::onSnapshotPoll()
{
    // Not strictly needed in headless batch scripts, but satisfies the linker
}

void HeadlessController::onSimulationProgress(int iteration, double error, double deltaError)
{
    Q_UNUSED(iteration);
    Q_UNUSED(error);
    Q_UNUSED(deltaError);
}

} // namespace salr