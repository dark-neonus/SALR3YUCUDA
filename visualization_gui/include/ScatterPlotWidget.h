/*
 * ScatterPlotWidget.h - OpenGL 3D scatter plot for density visualization
 */

#ifndef SALR_GUI_SCATTER_PLOT_WIDGET_H
#define SALR_GUI_SCATTER_PLOT_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QMatrix4x4>
#include <QVector3D>
#include <QColor>
#include "Types.h"

namespace salr {

class ScatterPlotWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT

public:
    explicit ScatterPlotWidget(QWidget* parent = nullptr);
    ~ScatterPlotWidget();

    void setSnapshotData(const SnapshotData& data);
    void setThreshold(double threshold);
    double threshold() const { return threshold_; }
    
    void setPointSize(float size);
    float pointSize() const { return pointSize_; }

    void setSpecies1Color(const QColor& color);
    void setSpecies2Color(const QColor& color);
    QColor species1Color() const { return species1Color_; }
    QColor species2Color() const { return species2Color_; }

    void resetView();

signals:
    void thresholdChanged(double threshold);
    void pointSizeChanged(float size);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void updateVertexData();
    void drawAxes();
    void updateProjection();

    SnapshotData currentData_;
    double threshold_ = 0.01;
    float pointSize_ = 500.0f;

    QColor species1Color_ = QColor(139, 0, 139);
    QColor species2Color_ = QColor(46, 139, 87);

    float rotationX_ = 30.0f;
    float rotationY_ = -45.0f;
    float zoom_ = 1.0f;
    QPoint lastMousePos_;
    
    int width_ = 1;
    int height_ = 1;
    bool needsProjectionUpdate_ = true;

    QOpenGLShaderProgram* shaderProgram_ = nullptr;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vboPositions_;
    QOpenGLBuffer vboColors_;
    int vertexCount_ = 0;

    QOpenGLVertexArrayObject axesVao_;
    QOpenGLBuffer axesVbo_;

    QMatrix4x4 projection_;
    QMatrix4x4 view_;
    QMatrix4x4 model_;

    float xMin_ = 0, xMax_ = 1;
    float yMin_ = 0, yMax_ = 1;
    float zMin_ = 0, zMax_ = 1;
};

} // namespace salr

#endif // SALR_GUI_SCATTER_PLOT_WIDGET_H
