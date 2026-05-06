/*
 * HeadlessController.h - Headless CLI controller for SALR Visualization
 */

#ifndef SALR_GUI_HEADLESS_CONTROLLER_H
#define SALR_GUI_HEADLESS_CONTROLLER_H

#include <QObject>
#include <QFile>
#include <QPair>
#include <QMap>
#include <QSet>
#include <QSocketNotifier>
#include <QTimer>
#include <QSize>
#include <initializer_list>

#include "DatabaseWrapper.h"
#include "SimulationRunner.h"
#include "VisualizationWidget.h"

namespace salr {

class HeadlessController : public QObject {
    Q_OBJECT

public:
    struct Options {
        QString databasePath;
        QString configPath;
        QString backend;
        QString sessionId;
        QSize renderSize;
    };

    explicit HeadlessController(const Options& options, QObject* parent = nullptr);

    void start();

private slots:
    void onStdinActivated();
    void onSimulationStarted(const QString& runId);
    void onSimulationProgress(int iteration, double error, double deltaError);
    void onSimulationFinished(bool converged, const QString& runId);
    void onSimulationError(const QString& message);
    void onSnapshotPoll();

private:
    void logEvent(const QString& event,
                  const std::initializer_list<QPair<QString, QString>>& fields = {});
    void logConfig(const SimulationConfig& config, const QString& backend, const QString& configPath);
    void startSimulation();

    bool openSession(const QString& runId);
    bool loadSnapshot(const QString& iterationToken);
    bool exportVisuals(const QString& path);

    Options options_;
    DatabaseWrapper database_;
    SimulationRunner runner_;
    VisualizationWidget* visualization_ = nullptr;

    QFile stdinFile_;
    QSocketNotifier* stdinNotifier_ = nullptr;
    QTimer snapshotTimer_;

    QString currentRunId_;
    int currentIteration_ = -1;
    QSet<int> knownSnapshots_;
    SimulationConfig currentConfig_;
    bool configLoaded_ = false;
    QString backendLabel_;
};

} // namespace salr

#endif // SALR_GUI_HEADLESS_CONTROLLER_H
