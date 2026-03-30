/*
 * RunControlWidget.h - Simulation launch control panel
 */

#ifndef SALR_GUI_RUN_CONTROL_WIDGET_H
#define SALR_GUI_RUN_CONTROL_WIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QGroupBox>
#include "Types.h"

namespace salr {

class DatabaseWrapper;

class RunControlWidget : public QWidget {
    Q_OBJECT

public:
    explicit RunControlWidget(DatabaseWrapper* database, QWidget* parent = nullptr);

    void setCurrentSession(const QString& runId);
    void setRunning(bool running);

signals:
    void startSimulation(const SimulationConfig& config, bool useCuda);
    void resumeSimulation(const QString& runId, int iteration, bool useCuda);
    void stopSimulation();

private slots:
    void onLoadDefault();
    void onLoadFromSession();
    void onStartCpu();
    void onStartCuda();
    void onResume();
    void onStop();

private:
    void setupUi();
    SimulationConfig buildConfig() const;
    void applyConfig(const SimulationConfig& config);

    DatabaseWrapper* database_;
    QString currentRunId_;
    bool isRunning_ = false;
    
    // Store potential parameters (not exposed in UI, but must be preserved)
    PotentialParams storedPotential_;

    // Grid parameters
    QSpinBox* nxSpin_ = nullptr;
    QSpinBox* nySpin_ = nullptr;
    QDoubleSpinBox* dxSpin_ = nullptr;
    QDoubleSpinBox* dySpin_ = nullptr;
    QComboBox* boundaryCombo_ = nullptr;

    // Physics parameters
    QDoubleSpinBox* tempSpin_ = nullptr;
    QDoubleSpinBox* rho1Spin_ = nullptr;
    QDoubleSpinBox* rho2Spin_ = nullptr;
    QDoubleSpinBox* cutoffSpin_ = nullptr;

    // Solver parameters
    QSpinBox* maxIterSpin_ = nullptr;
    QDoubleSpinBox* tolSpin_ = nullptr;
    QDoubleSpinBox* xi1Spin_ = nullptr;
    QDoubleSpinBox* xi2Spin_ = nullptr;
    QSpinBox* saveEverySpin_ = nullptr;

    // Resume options
    QCheckBox* resumeCheck_ = nullptr;
    QComboBox* snapshotCombo_ = nullptr;

    // Buttons
    QPushButton* loadDefaultBtn_ = nullptr;
    QPushButton* loadSessionBtn_ = nullptr;
    QPushButton* startCpuBtn_ = nullptr;
    QPushButton* startCudaBtn_ = nullptr;
    QPushButton* stopBtn_ = nullptr;
};

} // namespace salr

#endif // SALR_GUI_RUN_CONTROL_WIDGET_H
