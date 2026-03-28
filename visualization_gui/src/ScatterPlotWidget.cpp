/*
 * ScatterPlotWidget.cpp - OpenGL 3D scatter plot implementation
 */

#include "ScatterPlotWidget.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QtMath>
#include <QDebug>

namespace salr {

// Vertex shader source
static const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

uniform mat4 mvp;
uniform float pointSize;

out vec4 fragColor;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = pointSize;
    fragColor = color;
}
)";

// Fragment shader source
static const char* fragmentShaderSource = R"(
#version 330 core
in vec4 fragColor;
out vec4 outColor;

void main() {
    // Make points circular
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) {
        discard;
    }
    outColor = fragColor;
}
)";

ScatterPlotWidget::ScatterPlotWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , vboPositions_(QOpenGLBuffer::VertexBuffer)
    , vboColors_(QOpenGLBuffer::VertexBuffer)
    , axesVbo_(QOpenGLBuffer::VertexBuffer)
{
    setMinimumSize(200, 200);
}

ScatterPlotWidget::~ScatterPlotWidget()
{
    makeCurrent();

    if (shaderProgram_) {
        delete shaderProgram_;
    }

    vao_.destroy();
    vboPositions_.destroy();
    vboColors_.destroy();
    axesVao_.destroy();
    axesVbo_.destroy();

    doneCurrent();
}

void ScatterPlotWidget::setSnapshotData(const SnapshotData& data)
{
    currentData_ = data;

    if (data.isValid()) {
        xMin_ = 0;
        xMax_ = data.meta.Lx;
        yMin_ = 0;
        yMax_ = data.meta.Ly;
        zMin_ = 0;
        zMax_ = qMax(data.rho1Max, data.rho2Max) * 1.1;
    }

    if (isValid()) {
        makeCurrent();
        updateVertexData();
        doneCurrent();
        update();
    }
}

void ScatterPlotWidget::setThreshold(double threshold)
{
    if (qFuzzyCompare(threshold_, threshold)) return;

    threshold_ = threshold;

    if (isValid()) {
        makeCurrent();
        updateVertexData();
        doneCurrent();
        update();
    }

    emit thresholdChanged(threshold);
}

void ScatterPlotWidget::resetView()
{
    rotationX_ = 30.0f;
    rotationY_ = -45.0f;
    zoom_ = 1.0f;
    update();
}

void ScatterPlotWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Create shader program
    shaderProgram_ = new QOpenGLShaderProgram(this);
    if (!shaderProgram_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource)) {
        qWarning() << "Failed to compile vertex shader:" << shaderProgram_->log();
    }
    if (!shaderProgram_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource)) {
        qWarning() << "Failed to compile fragment shader:" << shaderProgram_->log();
    }
    if (!shaderProgram_->link()) {
        qWarning() << "Failed to link shader program:" << shaderProgram_->log();
    }

    // Create VAO and VBOs for scatter points
    vao_.create();
    vboPositions_.create();
    vboColors_.create();

    // Create VAO and VBO for axes
    axesVao_.create();
    axesVbo_.create();

    // Set up axes geometry
    axesVao_.bind();
    axesVbo_.bind();

    // Axes lines: X (red), Y (green), Z (blue)
    float axesData[] = {
        // X axis
        0, 0, 0,  1, 0, 0, 1,
        1, 0, 0,  1, 0, 0, 1,
        // Y axis
        0, 0, 0,  0, 1, 0, 1,
        0, 1, 0,  0, 1, 0, 1,
        // Z axis
        0, 0, 0,  0, 0, 1, 1,
        0, 0, 1,  0, 0, 1, 1,
    };
    axesVbo_.allocate(axesData, sizeof(axesData));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));

    axesVao_.release();
    axesVbo_.release();
}

void ScatterPlotWidget::resizeGL(int w, int h)
{
    float aspect = float(w) / float(h ? h : 1);
    projection_.setToIdentity();
    projection_.perspective(45.0f, aspect, 0.1f, 100.0f);
}

void ScatterPlotWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!shaderProgram_ || vertexCount_ == 0) {
        return;
    }

    shaderProgram_->bind();

    // Compute view matrix
    view_.setToIdentity();

    // Center the model
    float cx = (xMax_ + xMin_) / 2.0f;
    float cy = (yMax_ + yMin_) / 2.0f;
    float cz = (zMax_ + zMin_) / 2.0f;
    float maxDim = qMax(qMax(xMax_ - xMin_, yMax_ - yMin_), zMax_ - zMin_);
    float distance = maxDim * 2.0f / zoom_;

    view_.translate(0, 0, -distance);
    view_.rotate(rotationX_, 1, 0, 0);
    view_.rotate(rotationY_, 0, 0, 1);
    view_.translate(-cx, -cy, -cz);

    // Model matrix (identity)
    model_.setToIdentity();

    QMatrix4x4 mvp = projection_ * view_ * model_;
    shaderProgram_->setUniformValue("mvp", mvp);
    shaderProgram_->setUniformValue("pointSize", 4.0f);

    // Draw scatter points
    vao_.bind();
    glDrawArrays(GL_POINTS, 0, vertexCount_);
    vao_.release();

    // Draw axes (scaled to data bounds)
    QMatrix4x4 axesModel;
    axesModel.scale(xMax_ - xMin_, yMax_ - yMin_, zMax_ - zMin_);

    QMatrix4x4 axesMvp = projection_ * view_ * axesModel;
    shaderProgram_->setUniformValue("mvp", axesMvp);
    shaderProgram_->setUniformValue("pointSize", 1.0f);

    axesVao_.bind();
    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, 6);
    axesVao_.release();

    shaderProgram_->release();
}

void ScatterPlotWidget::updateVertexData()
{
    if (!currentData_.isValid()) {
        vertexCount_ = 0;
        return;
    }

    int nx = currentData_.meta.nx;
    int ny = currentData_.meta.ny;
    double dx = currentData_.meta.dx;
    double dy = currentData_.meta.dy;

    // Pre-count vertices above threshold
    int count = 0;
    for (int i = 0; i < nx * ny; ++i) {
        if (currentData_.rho1[i] > threshold_) count++;
        if (currentData_.rho2[i] > threshold_) count++;
    }

    if (count == 0) {
        vertexCount_ = 0;
        return;
    }

    // Build vertex data
    QVector<float> positions;
    QVector<float> colors;
    positions.reserve(count * 3);
    colors.reserve(count * 4);

    // Colors: species 1 = purple (0.545, 0, 0.545), species 2 = green (0.18, 0.545, 0.34)
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int idx = iy * nx + ix;
            float x = (ix + 0.5f) * dx;
            float y = (iy + 0.5f) * dy;

            // Species 1 (purple)
            double r1 = currentData_.rho1[idx];
            if (r1 > threshold_) {
                positions << x << y << static_cast<float>(r1);
                colors << 0.545f << 0.0f << 0.545f << 0.7f;
            }

            // Species 2 (green)
            double r2 = currentData_.rho2[idx];
            if (r2 > threshold_) {
                positions << x << y << static_cast<float>(r2);
                colors << 0.18f << 0.545f << 0.34f << 0.7f;
            }
        }
    }

    vertexCount_ = positions.size() / 3;

    // Upload to GPU
    vao_.bind();

    vboPositions_.bind();
    vboPositions_.allocate(positions.constData(), positions.size() * sizeof(float));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    vboPositions_.release();

    vboColors_.bind();
    vboColors_.allocate(colors.constData(), colors.size() * sizeof(float));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    vboColors_.release();

    vao_.release();
}

void ScatterPlotWidget::mousePressEvent(QMouseEvent* event)
{
    lastMousePos_ = event->pos();
}

void ScatterPlotWidget::mouseMoveEvent(QMouseEvent* event)
{
    int dx = event->pos().x() - lastMousePos_.x();
    int dy = event->pos().y() - lastMousePos_.y();

    if (event->buttons() & Qt::LeftButton) {
        rotationX_ += dy * 0.5f;
        rotationY_ += dx * 0.5f;

        // Clamp rotation
        rotationX_ = qBound(-90.0f, rotationX_, 90.0f);

        update();
    }

    lastMousePos_ = event->pos();
}

void ScatterPlotWidget::wheelEvent(QWheelEvent* event)
{
    float delta = event->angleDelta().y() / 120.0f;
    zoom_ *= qPow(1.1f, delta);
    zoom_ = qBound(0.1f, zoom_, 10.0f);
    update();
}

} // namespace salr
