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
#include <QProcessEnvironment>

namespace salr {

SimulationRunner::SimulationRunner(QObject* parent)
    : QObject(parent)
{
    process_ = new QProcess(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);

    connect(process_, &QProcess::started, this, &SimulationRunner::onProcessStarted);
    connect(process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &SimulationRunner::onProcessFinished);
    connect(process_, &QProcess::errorOccurred, this, &SimulationRunner::onProcessError);
    
    // Connect to the formally declared slot
    connect(process_, &QProcess::readyReadStandardOutput, this, &SimulationRunner::onProcessReadyRead);
}

SimulationRunner::~SimulationRunner()
{
    stop();
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
    lastConverged_ = false;

    tempConfigPath_ = QDir::temp().filePath(
        QString("salr_config_%1.cfg").arg(QDateTime::currentMSecsSinceEpoch()));

    if (!writeConfigFile(config, tempConfigPath_)) {
        emit errorOccurred(tr("Failed to write config file"));
        return;
    }

    QString executable = findExecutable(useCuda);
    if (executable.isEmpty()) {
        emit errorOccurred(tr("Simulation executable not found"));
        return;
    }

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QString exeDir = QFileInfo(executable).absolutePath();
    QString currentLd = env.value("LD_LIBRARY_PATH");
    env.insert("LD_LIBRARY_PATH", currentLd.isEmpty() ? exeDir : currentLd + ":" + exeDir);
    process_->setProcessEnvironment(env);
    process_->setWorkingDirectory(QDir::currentPath());

    QStringList args;
    args << tempConfigPath_;
    process_->start(executable, args);
}

void SimulationRunner::resume(const QString& runId, int iteration, const SimulationConfig& config, bool useCuda)
{
    if (isRunning()) {
        emit errorOccurred(tr("A simulation is already running"));
        return;
    }

    resumeRunId_ = runId;
    resumeIteration_ = iteration;
    lastConverged_ = false;

    tempConfigPath_ = QDir::temp().filePath(
        QString("salr_config_%1.cfg").arg(QDateTime::currentMSecsSinceEpoch()));

    if (!writeConfigFile(config, tempConfigPath_)) {
        emit errorOccurred(tr("Failed to write config file"));
        return;
    }

    QString executable = findExecutable(useCuda);
    if (executable.isEmpty()) {
        emit errorOccurred(tr("Simulation executable not found"));
        return;
    }

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QString exeDir = QFileInfo(executable).absolutePath();
    QString currentLd = env.value("LD_LIBRARY_PATH");
    env.insert("LD_LIBRARY_PATH", currentLd.isEmpty() ? exeDir : currentLd + ":" + exeDir);
    process_->setProcessEnvironment(env);
    process_->setWorkingDirectory(QDir::currentPath());

    QStringList args;
    args << tempConfigPath_ << "--resume" << runId << QString::number(iteration);
    process_->start(executable, args);
}

void SimulationRunner::stop()
{
    if (isRunning()) {
        process_->terminate();
        if (!process_->waitForFinished(3000)) {
            process_->kill();
        }
    }
}

bool SimulationRunner::isRunning() const
{
    return process_ && process_->state() != QProcess::NotRunning;
}

QString SimulationRunner::findExecutable(bool useCuda) const
{
    QString exeName = useCuda ? "salr_dft_cuda_db" : "salr_dft_db";
#ifdef Q_OS_WIN
    exeName += ".exe";
#endif

    if (!executablePath_.isEmpty() && QFile::exists(executablePath_)) {
        return executablePath_;
    }

    QStringList searchPaths = {
        QCoreApplication::applicationDirPath() + "/" + exeName,
        QCoreApplication::applicationDirPath() + "/../" + exeName,
        QCoreApplication::applicationDirPath() + "/../build/" + exeName,
        QDir::currentPath() + "/build/" + exeName,
        QDir::currentPath() + "/" + exeName
    };

    for (const QString& path : searchPaths) {
        if (QFile::exists(path)) {
            return path;
        }
    }
    return QString();
}

bool SimulationRunner::writeConfigFile(const SimulationConfig& config, const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream out(&file);

    out << "[grid]\n";
    out << "nx = " << config.grid.nx << "\n";
    out << "ny = " << config.grid.ny << "\n";
    out << "dx = " << config.grid.dx << "\n";
    out << "dy = " << config.grid.dy << "\n";
    
    QString bcStr = "PBC";
    if (config.boundaryMode == BoundaryMode::W2) bcStr = "W2";
    else if (config.boundaryMode == BoundaryMode::W4) bcStr = "W4";
    out << "boundary_mode = " << bcStr << "\n";
    out << "init_mode = " << config.initMode << "\n\n";

    out << "[physics]\n";
    out << "temperature = " << config.temperature << "\n";
    out << "rho1 = " << config.rho1 << "\n";
    out << "rho2 = " << config.rho2 << "\n";
    out << "cutoff_radius = " << config.potential.cutoffRadius << "\n\n";

    out << "[interaction]\n";
    for (int i = 0; i < 3; ++i) {
        out << "A_11_" << (i+1) << " = " << config.potential.A[0][0][i] << "\n";
        out << "a_11_" << (i+1) << " = " << config.potential.alpha[0][0][i] << "\n";
    }
    for (int i = 0; i < 3; ++i) {
        out << "A_12_" << (i+1) << " = " << config.potential.A[0][1][i] << "\n";
        out << "a_12_" << (i+1) << " = " << config.potential.alpha[0][1][i] << "\n";
    }
    for (int i = 0; i < 3; ++i) {
        out << "A_22_" << (i+1) << " = " << config.potential.A[1][1][i] << "\n";
        out << "a_22_" << (i+1) << " = " << config.potential.alpha[1][1][i] << "\n";
    }
    out << "\n";

    out << "[solver]\n";
    out << "max_iterations = " << config.solver.maxIterations << "\n";
    out << "tolerance = " << config.solver.tolerance << "\n";
    out << "xi1 = " << config.solver.xi1 << "\n";
    out << "xi2 = " << config.solver.xi2 << "\n\n";

    out << "[output]\n";
    out << "output_dir = output/\n";
    out << "save_every = " << config.saveEvery << "\n";
    if (!databasePath_.isEmpty()) {
        out << "database_path = " << databasePath_ << "\n";
    }

    return true;
}

void SimulationRunner::onProcessStarted()
{
}

void SimulationRunner::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::CrashExit) {
        emit errorOccurred(tr("Simulation process crashed (exit code %1)").arg(exitCode));
    }
    emit finished(lastConverged_, currentRunId_);
}

void SimulationRunner::onProcessError(QProcess::ProcessError error)
{
    if (error == QProcess::FailedToStart) {
        emit errorOccurred(tr("Failed to start simulation executable"));
    }
}

// Re-added to satisfy MOC/Linker
void SimulationRunner::onProcessReadyRead()
{
    while (process_->canReadLine()) {
        QString line = QString::fromLocal8Bit(process_->readLine()).trimmed();
        if (line.isEmpty()) continue;

        QTextStream(stdout) << "  [SOLVER] " << line << Qt::endl;

        // UPDATED REGEX: Look for "Created run:"
        static QRegularExpression runIdRe("Created run:\\s*([a-zA-Z0-9_-]+)");
        auto match = runIdRe.match(line);
        if (match.hasMatch()) {
            currentRunId_ = match.captured(1);
            emit started(currentRunId_);
        }

        static QRegularExpression nonConvRe(R"((did\s+not\s+converge|not\s+converged|failed\s+to\s+converge))",
                                            QRegularExpression::CaseInsensitiveOption);
        static QRegularExpression convRe(R"((\bconverged\b|convergence\s+reached))",
                                         QRegularExpression::CaseInsensitiveOption);

        if (nonConvRe.match(line).hasMatch()) {
            lastConverged_ = false;
        } else if (convRe.match(line).hasMatch()) {
            lastConverged_ = true;
        }
    }
}

} // namespace salr