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
#include "Types.h"

namespace salr {

class HeatmapWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT

public:
    enum ColorMode {
        Clipped,    // Clipped to 3x mean
        AutoScale   // Full range
    };

    explicit HeatmapWidget(QWidget* parent = nullptr);
    ~HeatmapWidget();

    void setSnapshotData(const SnapshotData& data);

    void setColorMode(ColorMode mode);
    ColorMode colorMode() const { return colorMode_; }

signals:
    void cursorPosition(double x, double y, double rho1, double rho2);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mouseMoveEvent(QMouseEvent* event) override;

private:
    void updateTexture();
    void drawColorbar();

    // Data
    SnapshotData currentData_;
    ColorMode colorMode_ = Clipped;

    // OpenGL objects
    QOpenGLShaderProgram* shaderProgram_ = nullptr;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_;
    QOpenGLTexture* texture_ = nullptr;

    // Viewport
    int viewportWidth_ = 0;
    int viewportHeight_ = 0;
};

} // namespace salr

#endif // SALR_GUI_HEATMAP_WIDGET_H
