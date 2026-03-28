/*
 * SimulationRunner.cpp - External process management implementation
 */

#include "SimulationRunner.h"
#include "DatabaseWrapper.h"

#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>

namespace salr {

SimulationRunner::SimulationRunner(QObject* parent)
    : QObject(parent)
{
}

SimulationRunner::~SimulationRunner()
{
    stop();

    // Clean up temp config file
    if (!tempConfigPath_.isEmpty() && QFile::exists(tempConfigPath_)) {
        QFile::remove(tempConfigPath_);
    }
}

void SimulationRunner::startNew(const SimulationConfig& config, bool useCuda)
{
    if (isRunning()) {
        emit errorOccurred(tr("A simulation is already running"));
        return;
    }

    resumeRunId_.clear();
    resumeIteration_ = -1;

    // Generate temp config file
    tempConfigPath_ = QDir::temp().filePath(
        QString("salr_config_%1.cfg").arg(QDateTime::currentMSecsSinceEpoch()));

    if (!writeConfigFile(config, tempConfigPath_)) {
        emit errorOccurred(tr("Failed to write config file"));
        return;
    }

    // Find executable
    QString executable = findExecutable(useCuda);
    if (executable.isEmpty()) {
        emit errorOccurred(tr("Simulation executable not found"));
        return;
    }

    // Build command
    QStringList args;
    args << tempConfigPath_;

    // Setup process
    process_ = new QProcess(this);

    connect(process_, &QProcess::started,
            this, &SimulationRunner::onProcessStarted);
    connect(process_, &QProcess::readyReadStandardOutput,
            this, &SimulationRunner::onProcessReadyRead);
    connect(process_, &QProcess::readyReadStandardError,
            this, &SimulationRunner::onProcessReadyRead);
    connect(process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &SimulationRunner::onProcessFinished);
    connect(process_, &QProcess::errorOccurred,
            this, &SimulationRunner::onProcessError);

    // Set environment
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (!databasePath_.isEmpty()) {
        env.insert("SALR_DB_PATH", databasePath_);
    }
    process_->setProcessEnvironment(env);

    // Set working directory
    process_->setWorkingDirectory(QDir::currentPath());

    qDebug() << "Starting simulation:" << executable << args;
    process_->start(executable, args);
}

void SimulationRunner::resume(const QString& runId, int iteration,
                               const SimulationConfig& config, bool useCuda)
{
    if (isRunning()) {
        emit errorOccurred(tr("A simulation is already running"));
        return;
    }

    resumeRunId_ = runId;
    resumeIteration_ = iteration;

    // Generate temp config file
    tempConfigPath_ = QDir::temp().filePath(
        QString("salr_config_%1.cfg").arg(QDateTime::currentMSecsSinceEpoch()));

    if (!writeConfigFile(config, tempConfigPath_)) {
        emit errorOccurred(tr("Failed to write config file"));
        return;
    }

    // Find executable
    QString executable = findExecutable(useCuda);
    if (executable.isEmpty()) {
        emit errorOccurred(tr("Simulation executable not found"));
        return;
    }

    // Build command with resume args
    QStringList args;
    args << tempConfigPath_;
    args << "--resume" << runId;
    if (iteration >= 0) {
        args << QString::number(iteration);
    }

    // Setup process
    process_ = new QProcess(this);

    connect(process_, &QProcess::started,
            this, &SimulationRunner::onProcessStarted);
    connect(process_, &QProcess::readyReadStandardOutput,
            this, &SimulationRunner::onProcessReadyRead);
    connect(process_, &QProcess::readyReadStandardError,
            this, &SimulationRunner::onProcessReadyRead);
    connect(process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &SimulationRunner::onProcessFinished);
    connect(process_, &QProcess::errorOccurred,
            this, &SimulationRunner::onProcessError);

    // Set environment
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (!databasePath_.isEmpty()) {
        env.insert("SALR_DB_PATH", databasePath_);
    }
    process_->setProcessEnvironment(env);

    qDebug() << "Resuming simulation:" << executable << args;
    process_->start(executable, args);
}

void SimulationRunner::stop()
{
    if (process_ && process_->state() != QProcess::NotRunning) {
        process_->terminate();

        if (!process_->waitForFinished(3000)) {
            process_->kill();
            process_->waitForFinished(1000);
        }
    }

    if (process_) {
        process_->deleteLater();
        process_ = nullptr;
    }
}

bool SimulationRunner::isRunning() const
{
    return process_ && process_->state() != QProcess::NotRunning;
}

void SimulationRunner::onProcessStarted()
{
    qDebug() << "Simulation process started";

    // We don't know the run ID yet until we parse output
    // For resume, we use the known run ID
    if (!resumeRunId_.isEmpty()) {
        currentRunId_ = resumeRunId_;
    } else {
        currentRunId_ = "starting...";
    }

    emit started(currentRunId_);
}

void SimulationRunner::onProcessReadyRead()
{
    if (!process_) return;

    // Read stdout
    while (process_->canReadLine()) {
        QByteArray line = process_->readLine();
        QString text = QString::fromUtf8(line).trimmed();
        if (!text.isEmpty()) {
            emit outputLine(text);
            parseOutputLine(text);
        }
    }

    // Also read any remaining data
    QByteArray remaining = process_->readAllStandardOutput();
    if (!remaining.isEmpty()) {
        for (const QString& line : QString::fromUtf8(remaining).split('\n')) {
            QString text = line.trimmed();
            if (!text.isEmpty()) {
                emit outputLine(text);
                parseOutputLine(text);
            }
        }
    }

    // Read stderr
    remaining = process_->readAllStandardError();
    if (!remaining.isEmpty()) {
        for (const QString& line : QString::fromUtf8(remaining).split('\n')) {
            QString text = line.trimmed();
            if (!text.isEmpty()) {
                emit outputLine("[stderr] " + text);
            }
        }
    }
}

void SimulationRunner::onProcessFinished(int exitCode, QProcess::ExitStatus status)
{
    qDebug() << "Simulation finished with code" << exitCode << "status" << status;

    // Clean up temp file
    if (!tempConfigPath_.isEmpty() && QFile::exists(tempConfigPath_)) {
        QFile::remove(tempConfigPath_);
        tempConfigPath_.clear();
    }

    bool success = (status == QProcess::NormalExit && exitCode == 0);
    emit finished(lastConverged_, currentRunId_);

    process_->deleteLater();
    process_ = nullptr;
    currentRunId_.clear();
}

void SimulationRunner::onProcessError(QProcess::ProcessError error)
{
    QString errorMsg;
    switch (error) {
        case QProcess::FailedToStart:
            errorMsg = tr("Failed to start simulation process");
            break;
        case QProcess::Crashed:
            errorMsg = tr("Simulation process crashed");
            break;
        case QProcess::Timedout:
            errorMsg = tr("Simulation process timed out");
            break;
        case QProcess::WriteError:
            errorMsg = tr("Error writing to simulation process");
            break;
        case QProcess::ReadError:
            errorMsg = tr("Error reading from simulation process");
            break;
        default:
            errorMsg = tr("Unknown process error");
            break;
    }

    emit errorOccurred(errorMsg);
}

bool SimulationRunner::writeConfigFile(const SimulationConfig& config, const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream out(&file);
    out.setRealNumberPrecision(10);

    // Grid section
    out << "[grid]\n";
    out << "dx = " << config.grid.dx << "\n";
    out << "dy = " << config.grid.dy << "\n";
    out << "nx = " << config.grid.nx << "\n";
    out << "ny = " << config.grid.ny << "\n";
    out << "boundary_mode = " << boundaryModeToString(config.boundaryMode) << "\n";
    out << "\n";

    // Physics section
    out << "[physics]\n";
    out << "temperature = " << config.temperature << "\n";
    out << "rho1 = " << config.rho1 << "\n";
    out << "rho2 = " << config.rho2 << "\n";
    out << "cutoff_radius = " << config.potential.cutoffRadius << "\n";
    out << "\n";

    // Interaction section (use defaults if not set)
    out << "[interaction]\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = i; j < 2; ++j) {
            for (int m = 0; m < 3; ++m) {
                out << QString("A_%1%2_%3 = ").arg(i+1).arg(j+1).arg(m+1)
                    << config.potential.A[i][j][m] << "\n";
            }
            for (int m = 0; m < 3; ++m) {
                out << QString("a_%1%2_%3 = ").arg(i+1).arg(j+1).arg(m+1)
                    << config.potential.alpha[i][j][m] << "\n";
            }
        }
    }
    out << "\n";

    // Solver section
    out << "[solver]\n";
    out << "max_iterations = " << config.solver.maxIterations << "\n";
    out << "tolerance = " << config.solver.tolerance << "\n";
    out << "xi1 = " << config.solver.xi1 << "\n";
    out << "xi2 = " << config.solver.xi2 << "\n";
    out << "error_change_threshold = " << config.solver.errorChangeThreshold << "\n";
    out << "xi_damping_factor = " << config.solver.xiDampingFactor << "\n";
    out << "\n";

    // Output section
    out << "[output]\n";
    out << "output_dir = " << config.outputDir << "\n";
    out << "save_every = " << config.saveEvery << "\n";

    return true;
}

void SimulationRunner::parseOutputLine(const QString& line)
{
    // Parse run ID from "Session: session_YYYYMMDD_HHMMSS_hash"
    static QRegularExpression sessionRe(R"(Session:\s*(\S+))");
    QRegularExpressionMatch sessionMatch = sessionRe.match(line);
    if (sessionMatch.hasMatch()) {
        currentRunId_ = sessionMatch.captured(1);
        emit started(currentRunId_);
    }

    // Parse iteration progress
    // Format: "Iteration NNNN: error = X.XXXe-YY, delta = X.XXXe-YY"
    static QRegularExpression iterRe(R"(Iteration\s+(\d+):\s+error\s*=\s*([\d.eE+-]+)(?:,\s*delta\s*=\s*([\d.eE+-]+))?)");
    QRegularExpressionMatch iterMatch = iterRe.match(line);
    if (iterMatch.hasMatch()) {
        int iteration = iterMatch.captured(1).toInt();
        double error = iterMatch.captured(2).toDouble();
        double delta = iterMatch.captured(3).isEmpty() ? 0.0 : iterMatch.captured(3).toDouble();
        emit progress(iteration, error, delta);
    }

    // Parse convergence
    if (line.contains("converged", Qt::CaseInsensitive) ||
        line.contains("Convergence reached", Qt::CaseInsensitive)) {
        lastConverged_ = true;
    }
}

QString SimulationRunner::findExecutable(bool useCuda) const
{
    QString exeName = useCuda ? "salr_dft_cuda_db" : "salr_dft_db";
#ifdef Q_OS_WIN
    exeName += ".exe";
#endif

    // If executable path is set, use it
    if (!executablePath_.isEmpty() && QFile::exists(executablePath_)) {
        return executablePath_;
    }

    // Search in common locations
    QStringList searchPaths = {
        QCoreApplication::applicationDirPath() + "/" + exeName,
        QCoreApplication::applicationDirPath() + "/../" + exeName,
        QCoreApplication::applicationDirPath() + "/../build/" + exeName,
        QDir::currentPath() + "/build/" + exeName,
        QDir::currentPath() + "/" + exeName
    };

    for (const QString& path : searchPaths) {
        if (QFile::exists(path)) {
            return QDir::cleanPath(path);
        }
    }

    // Return just the name and hope it's in PATH
    return exeName;
}

} // namespace salr
