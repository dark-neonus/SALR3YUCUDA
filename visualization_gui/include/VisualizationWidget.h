/*
 * VisualizationWidget.h - Container for visualization tabs
 */

#ifndef SALR_GUI_VISUALIZATION_WIDGET_H
#define SALR_GUI_VISUALIZATION_WIDGET_H

#include <QWidget>
#include <QTabWidget>
#include <QSlider>
#include <QComboBox>
#include <QLabel>
#include "Types.h"

namespace salr {

class ScatterPlotWidget;
class HeatmapWidget;

class VisualizationWidget : public QWidget {
    Q_OBJECT

public:
    explicit VisualizationWidget(QWidget* parent = nullptr);

    void setSnapshotData(const SnapshotData& data);

private slots:
    void onThresholdChanged(int value);
    void onPointSizeChanged(int value);
    void onColorModeChanged(int index);
    void onCursorPosition(double x, double y, double rho1, double rho2);

private:
    void setupUi();

    QTabWidget* tabWidget_ = nullptr;
    ScatterPlotWidget* scatterPlot_ = nullptr;
    HeatmapWidget* heatmap_ = nullptr;

    // Controls
    QSlider* thresholdSlider_ = nullptr;
    QLabel* thresholdLabel_ = nullptr;
    QSlider* pointSizeSlider_ = nullptr;
    QLabel* pointSizeLabel_ = nullptr;
    QComboBox* colorModeCombo_ = nullptr;
    QLabel* cursorLabel_ = nullptr;
};

} // namespace salr

#endif // SALR_GUI_VISUALIZATION_WIDGET_H
