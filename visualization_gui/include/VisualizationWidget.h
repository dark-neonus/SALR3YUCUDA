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
#include <QPushButton>
#include <QColor>
#include "Types.h"

namespace salr {

class ScatterPlotWidget;
class HeatmapWidget;

class VisualizationWidget : public QWidget {
    Q_OBJECT

public:
    explicit VisualizationWidget(QWidget* parent = nullptr);

    void setSnapshotData(const SnapshotData& data);

    QColor species1Color() const { return species1Color_; }
    QColor species2Color() const { return species2Color_; }

signals:
    void species1ColorChanged(const QColor& color);
    void species2ColorChanged(const QColor& color);

private slots:
    void onThresholdChanged(int value);
    void onPointSizeChanged(int value);
    void onColorModeChanged(int index);
    void onCursorPosition(double x, double y, double rho1, double rho2);
    void onSpecies1ColorClicked();
    void onSpecies2ColorClicked();
    void onAxisScaleChanged(int index);

private:
    void setupUi();
    void updateColorButton(QPushButton* btn, const QColor& color);

    QTabWidget* tabWidget_ = nullptr;
    ScatterPlotWidget* scatterPlot_ = nullptr;
    HeatmapWidget* heatmap_ = nullptr;

    // Controls
    QSlider* thresholdSlider_ = nullptr;
    QLabel* thresholdLabel_ = nullptr;
    QSlider* pointSizeSlider_ = nullptr;
    QLabel* pointSizeLabel_ = nullptr;
    QComboBox* colorModeCombo_ = nullptr;
    QComboBox* axisScaleCombo_ = nullptr;
    QPushButton* species1ColorBtn_ = nullptr;
    QPushButton* species2ColorBtn_ = nullptr;
    QLabel* cursorLabel_ = nullptr;

    // Colors
    QColor species1Color_ = QColor(139, 0, 139);   // Purple
    QColor species2Color_ = QColor(46, 139, 87);   // SeaGreen
};

} // namespace salr

#endif // SALR_GUI_VISUALIZATION_WIDGET_H
