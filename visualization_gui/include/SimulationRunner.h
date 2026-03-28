/*
 * SimulationRunner.h - External process management for simulations
 */

#ifndef SALR_GUI_SIMULATION_RUNNER_H
#define SALR_GUI_SIMULATION_RUNNER_H

#include <QObject>
#include <QProcess>
#include <QString>
#include "Types.h"

namespace salr {

class SimulationRunner : public QObject {
    Q_OBJECT

public:
    explicit SimulationRunner(QObject* parent = nullptr);
    ~SimulationRunner();

    void setDatabasePath(const QString& path) { databasePath_ = path; }
    void setExecutablePath(const QString& path) { executablePath_ = path; }

    void startNew(const SimulationConfig& config, bool useCuda);
    void resume(const QString& runId, int iteration, const SimulationConfig& config, bool useCuda);
    void stop();

    bool isRunning() const;
    QString currentRunId() const { return currentRunId_; }

signals:
    void started(const QString& runId);
    void progress(int iteration, double error, double deltaError);
    void finished(bool converged, const QString& runId);
    void outputLine(const QString& line);
    void errorOccurred(const QString& message);

private slots:
    void onProcessStarted();
    void onProcessReadyRead();
    void onProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onProcessError(QProcess::ProcessError error);

private:
    bool writeConfigFile(const SimulationConfig& config, const QString& path);
    void parseOutputLine(const QString& line);
    QString findExecutable(bool useCuda) const;

    QProcess* process_ = nullptr;
    QString databasePath_;
    QString executablePath_;
    QString tempConfigPath_;
    QString currentRunId_;
    QString resumeRunId_;
    int resumeIteration_ = -1;
    bool lastConverged_ = false;
};

} // namespace salr

#endif // SALR_GUI_SIMULATION_RUNNER_H
