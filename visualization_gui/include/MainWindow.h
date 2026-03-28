/*
 * MainWindow.h - Main application window for SALR Visualization GUI
 */

#ifndef SALR_GUI_MAIN_WINDOW_H
#define SALR_GUI_MAIN_WINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <QDockWidget>
#include <QStatusBar>
#include <QProgressBar>
#include <QLabel>
#include <memory>

#include "Types.h"

namespace salr {

// Forward declarations
class DatabaseWrapper;
class SessionBrowserWidget;
class SnapshotBrowserWidget;
class VisualizationWidget;
class ParameterDisplayWidget;
class RunControlWidget;
class SimulationRunner;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    // Menu actions
    void onOpenDatabase();
    void onExportSnapshot();
    void onQuit();

    // Session browser interactions
    void onSessionSelected(const QString& runId);
    void onSessionDoubleClicked(const QString& runId);

    // Snapshot browser interactions
    void onSnapshotSelected(int iteration);
    void onSnapshotDoubleClicked(int iteration);

    // Run control
    void onStartSimulation(const SimulationConfig& config, bool useCuda);
    void onResumeSimulation(const QString& runId, int iteration, bool useCuda);
    void onStopSimulation();

    // Simulation runner signals
    void onSimulationStarted(const QString& runId);
    void onSimulationProgress(int iteration, double error, double deltaError);
    void onSimulationFinished(bool converged, const QString& runId);
    void onSimulationError(const QString& message);

    // Refresh
    void refreshSessionList();
    void refreshSnapshotList();

private:
    void setupUi();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void setupConnections();

    void loadSettings();
    void saveSettings();

    void showMessage(const QString& message, int timeout = 3000);
    void showError(const QString& message);

    void loadSnapshot(const QString& runId, int iteration);

    // Widgets
    std::unique_ptr<DatabaseWrapper> database_;
    std::unique_ptr<SimulationRunner> simulationRunner_;

    SessionBrowserWidget* sessionBrowser_ = nullptr;
    SnapshotBrowserWidget* snapshotBrowser_ = nullptr;
    VisualizationWidget* visualization_ = nullptr;
    ParameterDisplayWidget* parameterDisplay_ = nullptr;
    RunControlWidget* runControl_ = nullptr;

    QSplitter* mainSplitter_ = nullptr;
    QSplitter* leftSplitter_ = nullptr;
    QDockWidget* rightDock_ = nullptr;

    // Status bar widgets
    QLabel* statusLabel_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel* sessionLabel_ = nullptr;

    // Current state
    QString currentRunId_;
    int currentIteration_ = -1;
    QString databasePath_;
};

} // namespace salr

#endif // SALR_GUI_MAIN_WINDOW_H
