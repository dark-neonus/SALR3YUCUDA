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

out VS_OUT {
    vec4 color;
} vs_out;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    vs_out.color = color;
}
)";

// Geometry shader - converts points to quads
static const char* geometryShaderSource = R"(
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform float pointSize;
uniform vec2 viewportSize;  // Screen size in pixels

in VS_OUT {
    vec4 color;
} gs_in[];

out vec2 texCoord;
out vec4 fragColor;

void main() {
    // Convert point size from pixels to NDC
    vec2 halfSize = vec2(pointSize) / viewportSize;
    
    vec4 center = gl_in[0].gl_Position;
    fragColor = gs_in[0].color;
    
    // Bottom-left
    gl_Position = center + vec4(-halfSize.x, -halfSize.y, 0.0, 0.0);
    texCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    // Bottom-right
    gl_Position = center + vec4(halfSize.x, -halfSize.y, 0.0, 0.0);
    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    
    // Top-left
    gl_Position = center + vec4(-halfSize.x, halfSize.y, 0.0, 0.0);
    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    
    // Top-right
    gl_Position = center + vec4(halfSize.x, halfSize.y, 0.0, 0.0);
    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
)";

// Fragment shader source
static const char* fragmentShaderSource = R"(
#version 330 core
in vec2 texCoord;
in vec4 fragColor;
out vec4 outColor;

void main() {
    // Make points circular
    vec2 coord = texCoord - vec2(0.5);
    float dist = dot(coord, coord);
    
    // Use dot product instead of length for efficiency
    // Discard pixels outside circle
    if (dist > 0.25) {  // 0.5^2 = 0.25
        outColor = vec4(0.0);  // Transparent instead of discard
    } else {
        // Smooth edges
        float alpha = 1.0 - 4.0 * dist;  // Linear falloff from center
        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
    }
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
        
        // Bounds changed, need to update projection
        needsProjectionUpdate_ = true;
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

void ScatterPlotWidget::setPointSize(float size)
{
    if (qFuzzyCompare(pointSize_, size)) return;
    
    pointSize_ = qMax(100.0f, qMin(size, 1000.0f));
    update();
    emit pointSizeChanged(pointSize_);
}

void ScatterPlotWidget::setSpecies1Color(const QColor& color)
{
    if (species1Color_ == color) return;
    
    species1Color_ = color;
    
    if (isValid() && currentData_.isValid()) {
        makeCurrent();
        updateVertexData();
        doneCurrent();
        update();
    }
}

void ScatterPlotWidget::setSpecies2Color(const QColor& color)
{
    if (species2Color_ == color) return;
    
    species2Color_ = color;
    
    if (isValid() && currentData_.isValid()) {
        makeCurrent();
        updateVertexData();
        doneCurrent();
        update();
    }
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
    if (!shaderProgram_->addShaderFromSourceCode(QOpenGLShader::Geometry, geometryShaderSource)) {
        qWarning() << "Failed to compile geometry shader:" << shaderProgram_->log();
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

    // If data was already set before GL was initialized, update vertex data now
    if (currentData_.isValid()) {
        updateVertexData();
    }
}

void ScatterPlotWidget::resizeGL(int w, int h)
{
    width_ = w;
    height_ = h;
    needsProjectionUpdate_ = true;
}

void ScatterPlotWidget::updateProjection()
{
    float aspect = float(width_) / float(height_ ? height_ : 1);
    
    // Calculate scene-appropriate near/far planes
    float maxXY = qMax(xMax_ - xMin_, yMax_ - yMin_);
    if (maxXY < 0.1f) maxXY = 1.0f;  // Default if no data yet
    
    float fov = 45.0f * M_PI / 180.0f;
    float typicalDistance = maxXY / (2.0f * qTan(fov / 2.0f)) * 1.5f / zoom_;
    
    // Set near/far relative to viewing distance with good Z resolution
    float nearPlane = typicalDistance * 0.5f;   // Half the distance
    float farPlane = typicalDistance * 3.0f;     // 3x the distance
    
    projection_.setToIdentity();
    projection_.perspective(45.0f, aspect, nearPlane, farPlane);
}

void ScatterPlotWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!shaderProgram_ || vertexCount_ == 0) {
        return;
    }
    
    // Update projection if needed
    if (needsProjectionUpdate_) {
        updateProjection();
        needsProjectionUpdate_ = false;
    }

    shaderProgram_->bind();

    // Compute view matrix - simplified approach
    view_.setToIdentity();

    // Position camera further back to see the whole scene
    float cx = (xMax_ + xMin_) / 2.0f;
    float cy = (yMax_ + yMin_) / 2.0f;
    float maxXY = qMax(xMax_ - xMin_, yMax_ - yMin_);
    
    // Scale Z to make it visible (30% of XY extent)
    float zRange = zMax_ - zMin_;
    float zScale = (zRange > 0) ? (maxXY * 0.3f / zRange) : 1.0f;
    float scaledZRange = zRange * zScale;
    float cz = scaledZRange / 2.0f;  // Center of Z range after scaling
    
    // Camera distance should be enough to see the full scene
    // Use a simpler formula: distance based on FOV and scene size
    float fov = 45.0f * M_PI / 180.0f;  // Convert to radians
    float distance = maxXY / (2.0f * qTan(fov / 2.0f)) * 1.5f / zoom_;  // 1.5x for margin

    // Build view matrix: translate camera back, rotate, center on model
    view_.translate(0, 0, -distance);
    view_.rotate(rotationX_, 1, 0, 0);
    view_.rotate(rotationY_, 0, 1, 0);  // Rotate around Y instead of Z for more intuitive control
    view_.translate(-cx, -cy, -cz);

    // Model matrix - apply Z scaling only
    model_.setToIdentity();
    model_.scale(1.0f, 1.0f, zScale);

    QMatrix4x4 mvp = projection_ * view_ * model_;
    shaderProgram_->setUniformValue("mvp", mvp);
    shaderProgram_->setUniformValue("viewportSize", QVector2D(width_, height_));  // For geometry shader
    shaderProgram_->setUniformValue("pointSize", pointSize_);  // Use member variable

    // Draw scatter points
    vao_.bind();
    if (vertexCount_ > 0) {
        glDrawArrays(GL_POINTS, 0, vertexCount_);
    }
    vao_.release();

    // Draw axes (scaled to data bounds)
    QMatrix4x4 axesModel;
    axesModel.scale(xMax_ - xMin_, yMax_ - yMin_, scaledZRange);

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

    int count = 0;
    for (int i = 0; i < nx * ny; ++i) {
        if (currentData_.rho1[i] > threshold_) count++;
        if (currentData_.rho2[i] > threshold_) count++;
    }

    if (count == 0) {
        vertexCount_ = 0;
        return;
    }

    QVector<float> positions;
    QVector<float> colors;
    positions.reserve(count * 3);
    colors.reserve(count * 4);

    float c1r = species1Color_.redF();
    float c1g = species1Color_.greenF();
    float c1b = species1Color_.blueF();
    float c2r = species2Color_.redF();
    float c2g = species2Color_.greenF();
    float c2b = species2Color_.blueF();

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int idx = iy * nx + ix;
            float x = (ix + 0.5f) * dx;
            float y = (iy + 0.5f) * dy;

            double r1 = currentData_.rho1[idx];
            if (r1 > threshold_) {
                positions << x << y << static_cast<float>(r1);
                colors << c1r << c1g << c1b << 0.7f;
            }

            double r2 = currentData_.rho2[idx];
            if (r2 > threshold_) {
                positions << x << y << static_cast<float>(r2);
                colors << c2r << c2g << c2b << 0.7f;
            }
        }
    }

    vertexCount_ = positions.size() / 3;

    vao_.bind();

    vboPositions_.bind();
    vboPositions_.allocate(positions.constData(), positions.size() * sizeof(float));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

    vboColors_.bind();
    vboColors_.allocate(colors.constData(), colors.size() * sizeof(float));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);

    vboColors_.release();
    vao_.release();
    
    update();
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
