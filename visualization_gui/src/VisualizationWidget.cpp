/*
 * VisualizationWidget.cpp - Container for visualization tabs
 */

#include "VisualizationWidget.h"
#include "ScatterPlotWidget.h"
#include "HeatmapWidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QColorDialog>

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

    tabWidget_ = new QTabWidget();

    scatterPlot_ = new ScatterPlotWidget();
    heatmap_ = new HeatmapWidget();

    tabWidget_->addTab(scatterPlot_, tr("3D Scatter"));
    tabWidget_->addTab(heatmap_, tr("2D Heatmap"));

    layout->addWidget(tabWidget_, 1);

    QGroupBox* controlsGroup = new QGroupBox(tr("Visualization Controls"));
    QVBoxLayout* controlsMainLayout = new QVBoxLayout(controlsGroup);
    
    // First row: sliders and scale controls
    QHBoxLayout* row1Layout = new QHBoxLayout();

    // Threshold slider
    QLabel* threshLabel = new QLabel(tr("Threshold:"));
    row1Layout->addWidget(threshLabel);

    thresholdSlider_ = new QSlider(Qt::Horizontal);
    thresholdSlider_->setRange(0, 100);
    thresholdSlider_->setValue(1);
    thresholdSlider_->setMinimumWidth(100);
    thresholdSlider_->setMaximumWidth(200);
    thresholdSlider_->setMinimumHeight(20);
    row1Layout->addWidget(thresholdSlider_);

    thresholdLabel_ = new QLabel("0.01");
    thresholdLabel_->setMinimumWidth(50);
    row1Layout->addWidget(thresholdLabel_);

    row1Layout->addSpacing(20);

    // Point size slider
    QLabel* sizeLabel = new QLabel(tr("Point Size:"));
    row1Layout->addWidget(sizeLabel);

    pointSizeSlider_ = new QSlider(Qt::Horizontal);
    pointSizeSlider_->setRange(100, 1000);
    pointSizeSlider_->setValue(500);
    pointSizeSlider_->setMinimumWidth(100);
    pointSizeSlider_->setMaximumWidth(200);
    pointSizeSlider_->setMinimumHeight(20);
    row1Layout->addWidget(pointSizeSlider_);

    pointSizeLabel_ = new QLabel("500px");
    pointSizeLabel_->setMinimumWidth(50);
    row1Layout->addWidget(pointSizeLabel_);

    row1Layout->addSpacing(20);

    // Color mode combo
    QLabel* modeLabel = new QLabel(tr("Color Scale:"));
    row1Layout->addWidget(modeLabel);

    colorModeCombo_ = new QComboBox();
    colorModeCombo_->addItem(tr("Clipped (3x mean)"), static_cast<int>(HeatmapWidget::Clipped));
    colorModeCombo_->addItem(tr("Auto Scale"), static_cast<int>(HeatmapWidget::AutoScale));
    row1Layout->addWidget(colorModeCombo_);

    row1Layout->addSpacing(20);

    // Axis scale combo
    QLabel* axisLabel = new QLabel(tr("Axis Scale:"));
    row1Layout->addWidget(axisLabel);

    axisScaleCombo_ = new QComboBox();
    axisScaleCombo_->addItem(tr("Linear"), static_cast<int>(HeatmapWidget::Linear));
    axisScaleCombo_->addItem(tr("Logarithmic"), static_cast<int>(HeatmapWidget::Logarithmic));
    row1Layout->addWidget(axisScaleCombo_);

    row1Layout->addStretch();

    controlsMainLayout->addLayout(row1Layout);

    // Second row: color controls and cursor position
    QHBoxLayout* row2Layout = new QHBoxLayout();

    // Species color buttons
    QLabel* colorLabel = new QLabel(tr("Colors:"));
    row2Layout->addWidget(colorLabel);

    species1ColorBtn_ = new QPushButton(tr("Species 1"));
    species1ColorBtn_->setFixedWidth(80);
    updateColorButton(species1ColorBtn_, species1Color_);
    row2Layout->addWidget(species1ColorBtn_);

    species2ColorBtn_ = new QPushButton(tr("Species 2"));
    species2ColorBtn_->setFixedWidth(80);
    updateColorButton(species2ColorBtn_, species2Color_);
    row2Layout->addWidget(species2ColorBtn_);

    row2Layout->addSpacing(40);

    cursorLabel_ = new QLabel();
    cursorLabel_->setMinimumWidth(400);
    cursorLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    row2Layout->addWidget(cursorLabel_);

    row2Layout->addStretch();

    controlsMainLayout->addLayout(row2Layout);

    layout->addWidget(controlsGroup);

    // Connections
    connect(thresholdSlider_, &QSlider::valueChanged,
            this, &VisualizationWidget::onThresholdChanged);
    connect(pointSizeSlider_, &QSlider::valueChanged,
            this, &VisualizationWidget::onPointSizeChanged);
    connect(colorModeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &VisualizationWidget::onColorModeChanged);
    connect(axisScaleCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &VisualizationWidget::onAxisScaleChanged);
    connect(heatmap_, &HeatmapWidget::cursorPosition,
            this, &VisualizationWidget::onCursorPosition);
    connect(species1ColorBtn_, &QPushButton::clicked,
            this, &VisualizationWidget::onSpecies1ColorClicked);
    connect(species2ColorBtn_, &QPushButton::clicked,
            this, &VisualizationWidget::onSpecies2ColorClicked);
}

void VisualizationWidget::setSnapshotData(const SnapshotData& data)
{
    scatterPlot_->setSnapshotData(data);
    heatmap_->setSnapshotData(data);
}

void VisualizationWidget::onThresholdChanged(int value)
{
    double threshold = 0.001 * qPow(10.0, value / 33.3);
    threshold = qBound(0.001, threshold, 1.0);

    thresholdLabel_->setText(QString::number(threshold, 'g', 2));
    scatterPlot_->setThreshold(threshold);
}

void VisualizationWidget::onPointSizeChanged(int value)
{
    float size = static_cast<float>(value);
    pointSizeLabel_->setText(QString("%1px").arg(size, 0, 'f', 0));
    scatterPlot_->setPointSize(size);
}

void VisualizationWidget::onColorModeChanged(int index)
{
    HeatmapWidget::ColorMode mode = static_cast<HeatmapWidget::ColorMode>(
        colorModeCombo_->itemData(index).toInt());
    heatmap_->setColorMode(mode);
}

void VisualizationWidget::onAxisScaleChanged(int index)
{
    HeatmapWidget::AxisScale scale = static_cast<HeatmapWidget::AxisScale>(
        axisScaleCombo_->itemData(index).toInt());
    heatmap_->setAxisScale(scale);
}

void VisualizationWidget::onCursorPosition(double x, double y, double rho1, double rho2)
{
    cursorLabel_->setText(QString("x=%1, y=%2 | rho1=%3, rho2=%4")
        .arg(x, 0, 'f', 2).arg(y, 0, 'f', 2).arg(rho1, 0, 'f', 4).arg(rho2, 0, 'f', 4));
}

void VisualizationWidget::onSpecies1ColorClicked()
{
    QColor color = QColorDialog::getColor(species1Color_, this, tr("Select Species 1 Color"));
    if (color.isValid()) {
        species1Color_ = color;
        updateColorButton(species1ColorBtn_, color);
        heatmap_->setSpecies1Color(color);
        scatterPlot_->setSpecies1Color(color);
        emit species1ColorChanged(color);
    }
}

void VisualizationWidget::onSpecies2ColorClicked()
{
    QColor color = QColorDialog::getColor(species2Color_, this, tr("Select Species 2 Color"));
    if (color.isValid()) {
        species2Color_ = color;
        updateColorButton(species2ColorBtn_, color);
        heatmap_->setSpecies2Color(color);
        scatterPlot_->setSpecies2Color(color);
        emit species2ColorChanged(color);
    }
}

void VisualizationWidget::updateColorButton(QPushButton* btn, const QColor& color)
{
    QString style = QString("QPushButton { background-color: %1; color: %2; }")
        .arg(color.name())
        .arg(color.lightness() > 128 ? "black" : "white");
    btn->setStyleSheet(style);
}

} // namespace salr
