/*
 * VisualizationWidget.cpp - Container for visualization tabs
 */

#include "VisualizationWidget.h"
#include "ScatterPlotWidget.h"
#include "HeatmapWidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>

namespace salr {

VisualizationWidget::VisualizationWidget(QWidget* parent)
    : QWidget(parent)
{
    setupUi();
}

void VisualizationWidget::setupUi()
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    // Tab widget for different views
    tabWidget_ = new QTabWidget();

    scatterPlot_ = new ScatterPlotWidget();
    heatmap_ = new HeatmapWidget();

    tabWidget_->addTab(scatterPlot_, tr("3D Scatter"));
    tabWidget_->addTab(heatmap_, tr("2D Heatmap"));

    layout->addWidget(tabWidget_, 1);

    // Controls panel
    QGroupBox* controlsGroup = new QGroupBox(tr("Visualization Controls"));
    QHBoxLayout* controlsLayout = new QHBoxLayout(controlsGroup);

    // Threshold slider (for scatter plot)
    QLabel* threshLabel = new QLabel(tr("Threshold:"));
    controlsLayout->addWidget(threshLabel);

    thresholdSlider_ = new QSlider(Qt::Horizontal);
    thresholdSlider_->setRange(0, 100);
    thresholdSlider_->setValue(1);  // 1% default
    thresholdSlider_->setMaximumWidth(150);
    controlsLayout->addWidget(thresholdSlider_);

    thresholdLabel_ = new QLabel("0.01");
    thresholdLabel_->setMinimumWidth(50);
    controlsLayout->addWidget(thresholdLabel_);

    controlsLayout->addSpacing(20);

    // Color mode combo (for heatmap)
    QLabel* modeLabel = new QLabel(tr("Color Scale:"));
    controlsLayout->addWidget(modeLabel);

    colorModeCombo_ = new QComboBox();
    colorModeCombo_->addItem(tr("Clipped (3x mean)"), static_cast<int>(HeatmapWidget::Clipped));
    colorModeCombo_->addItem(tr("Auto Scale"), static_cast<int>(HeatmapWidget::AutoScale));
    controlsLayout->addWidget(colorModeCombo_);

    controlsLayout->addStretch();

    // Cursor position display
    cursorLabel_ = new QLabel();
    cursorLabel_->setMinimumWidth(250);
    controlsLayout->addWidget(cursorLabel_);

    layout->addWidget(controlsGroup);

    // Connections
    connect(thresholdSlider_, &QSlider::valueChanged,
            this, &VisualizationWidget::onThresholdChanged);
    connect(colorModeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &VisualizationWidget::onColorModeChanged);
    connect(heatmap_, &HeatmapWidget::cursorPosition,
            this, &VisualizationWidget::onCursorPosition);
}

void VisualizationWidget::setSnapshotData(const SnapshotData& data)
{
    scatterPlot_->setSnapshotData(data);
    heatmap_->setSnapshotData(data);
}

void VisualizationWidget::onThresholdChanged(int value)
{
    // Convert slider value (0-100) to threshold
    // Use logarithmic scale for better control
    double threshold = 0.001 * qPow(10.0, value / 33.3);  // ~0.001 to 1.0
    threshold = qBound(0.001, threshold, 1.0);

    thresholdLabel_->setText(QString::number(threshold, 'g', 2));
    scatterPlot_->setThreshold(threshold);
}

void VisualizationWidget::onColorModeChanged(int index)
{
    HeatmapWidget::ColorMode mode = static_cast<HeatmapWidget::ColorMode>(
        colorModeCombo_->itemData(index).toInt());
    heatmap_->setColorMode(mode);
}

void VisualizationWidget::onCursorPosition(double x, double y, double rho1, double rho2)
{
    cursorLabel_->setText(QString("x=%.2f, y=%.2f | rho1=%.4f, rho2=%.4f")
        .arg(x).arg(y).arg(rho1).arg(rho2));
}

} // namespace salr
