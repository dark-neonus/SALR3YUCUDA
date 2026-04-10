/*
 * HeatmapWidget.h - OpenGL 2D heatmap for density visualization
 */

#ifndef SALR_GUI_HEATMAP_WIDGET_H
#define SALR_GUI_HEATMAP_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QColor>
#include "Types.h"

namespace salr {

class HeatmapWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT

public:
    enum ColorMode {
        Clipped,
        AutoScale
    };

    enum AxisScale {
        Linear,
        Logarithmic
    };

    explicit HeatmapWidget(QWidget* parent = nullptr);
    ~HeatmapWidget();

    void setSnapshotData(const SnapshotData& data);

    void setColorMode(ColorMode mode);
    ColorMode colorMode() const { return colorMode_; }

    void setAxisScale(AxisScale scale);
    AxisScale axisScale() const { return axisScale_; }

    void setSpecies1Color(const QColor& color);
    void setSpecies2Color(const QColor& color);
    QColor species1Color() const { return species1Color_; }
    QColor species2Color() const { return species2Color_; }

signals:
    void cursorPosition(double x, double y, double rho1, double rho2);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mouseMoveEvent(QMouseEvent* event) override;

private:
    void updateTexture();

    SnapshotData currentData_;
    ColorMode colorMode_ = Clipped;
    AxisScale axisScale_ = Linear;

    QColor species1Color_ = QColor(139, 0, 139);
    QColor species2Color_ = QColor(46, 139, 87);

    QOpenGLShaderProgram* shaderProgram_ = nullptr;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_;
    QOpenGLTexture* texture_ = nullptr;

    int viewportWidth_ = 0;
    int viewportHeight_ = 0;
};

} // namespace salr

#endif // SALR_GUI_HEATMAP_WIDGET_H
