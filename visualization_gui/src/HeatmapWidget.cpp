/*
 * HeatmapWidget.cpp - OpenGL 2D heatmap implementation
 */

#include "HeatmapWidget.h"
#include <QMouseEvent>
#include <QtMath>

namespace salr {

static const char* heatmapVertexShader = R"(
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;

out vec2 fragTexCoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragTexCoord = texCoord;
}
)";

static const char* heatmapFragmentShader = R"(
#version 330 core
in vec2 fragTexCoord;
out vec4 outColor;

uniform sampler2D heatmapTexture;

void main() {
    outColor = texture(heatmapTexture, fragTexCoord);
}
)";

HeatmapWidget::HeatmapWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , vbo_(QOpenGLBuffer::VertexBuffer)
{
    setMinimumSize(200, 200);
    setMouseTracking(true);
}

HeatmapWidget::~HeatmapWidget()
{
    makeCurrent();

    if (shaderProgram_) {
        delete shaderProgram_;
    }
    if (texture_) {
        delete texture_;
    }

    vao_.destroy();
    vbo_.destroy();

    doneCurrent();
}

void HeatmapWidget::setSnapshotData(const SnapshotData& data)
{
    currentData_ = data;

    if (isValid()) {
        makeCurrent();
        updateTexture();
        doneCurrent();
        update();
    }
}

void HeatmapWidget::setColorMode(ColorMode mode)
{
    if (colorMode_ == mode) return;

    colorMode_ = mode;

    if (isValid() && currentData_.isValid()) {
        makeCurrent();
        updateTexture();
        doneCurrent();
        update();
    }
}

void HeatmapWidget::setAxisScale(AxisScale scale)
{
    if (axisScale_ == scale) return;

    axisScale_ = scale;

    if (isValid() && currentData_.isValid()) {
        makeCurrent();
        updateTexture();
        doneCurrent();
        update();
    }
}

void HeatmapWidget::setSpecies1Color(const QColor& color)
{
    if (species1Color_ == color) return;

    species1Color_ = color;

    if (isValid() && currentData_.isValid()) {
        makeCurrent();
        updateTexture();
        doneCurrent();
        update();
    }
}

void HeatmapWidget::setSpecies2Color(const QColor& color)
{
    if (species2Color_ == color) return;

    species2Color_ = color;

    if (isValid() && currentData_.isValid()) {
        makeCurrent();
        updateTexture();
        doneCurrent();
        update();
    }
}

void HeatmapWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    shaderProgram_ = new QOpenGLShaderProgram(this);
    shaderProgram_->addShaderFromSourceCode(QOpenGLShader::Vertex, heatmapVertexShader);
    shaderProgram_->addShaderFromSourceCode(QOpenGLShader::Fragment, heatmapFragmentShader);
    shaderProgram_->link();

    vao_.create();
    vbo_.create();

    vao_.bind();
    vbo_.bind();

    float quadVertices[] = {
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f,

        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f,
        -1.0f,  1.0f,   0.0f, 1.0f,
    };
    vbo_.allocate(quadVertices, sizeof(quadVertices));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    vao_.release();
    vbo_.release();

    texture_ = new QOpenGLTexture(QOpenGLTexture::Target2D);

    if (currentData_.isValid()) {
        updateTexture();
    }
}

void HeatmapWidget::resizeGL(int w, int h)
{
    viewportWidth_ = w;
    viewportHeight_ = h;
    glViewport(0, 0, w, h);
}

void HeatmapWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    if (!shaderProgram_ || !texture_ || !texture_->isCreated()) {
        return;
    }

    shaderProgram_->bind();

    float dataAspect = 1.0f;
    if (currentData_.isValid() && currentData_.meta.Ly > 0) {
        dataAspect = currentData_.meta.Lx / currentData_.meta.Ly;
    }

    float viewAspect = float(viewportWidth_) / float(viewportHeight_);
    float scaleX = 1.0f, scaleY = 1.0f;

    if (dataAspect > viewAspect) {
        scaleY = viewAspect / dataAspect;
    } else {
        scaleX = dataAspect / viewAspect;
    }

    vbo_.bind();
    float quadVertices[] = {
        -scaleX, -scaleY,   0.0f, 0.0f,
         scaleX, -scaleY,   1.0f, 0.0f,
         scaleX,  scaleY,   1.0f, 1.0f,

        -scaleX, -scaleY,   0.0f, 0.0f,
         scaleX,  scaleY,   1.0f, 1.0f,
        -scaleX,  scaleY,   0.0f, 1.0f,
    };
    vbo_.write(0, quadVertices, sizeof(quadVertices));
    vbo_.release();

    texture_->bind();
    shaderProgram_->setUniformValue("heatmapTexture", 0);

    vao_.bind();
    glDrawArrays(GL_TRIANGLES, 0, 6);
    vao_.release();

    texture_->release();
    shaderProgram_->release();
}

void HeatmapWidget::updateTexture()
{
    if (!currentData_.isValid()) {
        return;
    }

    int nx = currentData_.meta.nx;
    int ny = currentData_.meta.ny;

    double rho1Min, rho1Max, rho2Min, rho2Max;
    if (colorMode_ == Clipped) {
        rho1Min = currentData_.rho1Min;
        rho2Min = currentData_.rho2Min;
        rho1Max = qMin(currentData_.rho1Max, 3.0 * currentData_.rho1Mean);
        rho2Max = qMin(currentData_.rho2Max, 3.0 * currentData_.rho2Mean);
    } else {
        rho1Min = currentData_.rho1Min;
        rho2Min = currentData_.rho2Min;
        rho1Max = currentData_.rho1Max;
        rho2Max = currentData_.rho2Max;
    }

    if (rho1Max < rho1Min) rho1Max = rho1Min;
    if (rho2Max < rho2Min) rho2Max = rho2Min;

    QVector<uchar> pixels(nx * ny * 4);

    double c1R = species1Color_.redF();
    double c1G = species1Color_.greenF();
    double c1B = species1Color_.blueF();
    double c2R = species2Color_.redF();
    double c2G = species2Color_.greenF();
    double c2B = species2Color_.blueF();

    const double eps = 1e-12;
    const double rho1Span = qMax(rho1Max - rho1Min, eps);
    const double rho2Span = qMax(rho2Max - rho2Min, eps);

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int srcIdx = iy * nx + ix;
            int dstIdx = ((ny - 1 - iy) * nx + ix) * 4;

            double r1 = currentData_.rho1[srcIdx];
            double r2 = currentData_.rho2[srcIdx];

            double n1, n2;
            if (axisScale_ == Logarithmic) {
                double logMin1 = qMax(1e-12, qMin(rho1Min, rho1Max));
                double logMin2 = qMax(1e-12, qMin(rho2Min, rho2Max));
                double logMax1 = qMax(logMin1 + 1e-12, rho1Max);
                double logMax2 = qMax(logMin2 + 1e-12, rho2Max);

                double den1 = qLn(logMax1 / logMin1);
                double den2 = qLn(logMax2 / logMin2);

                if (den1 > 1e-12) {
                    n1 = qBound(0.0, qLn(qMax(r1, logMin1) / logMin1) / den1, 1.0);
                } else {
                    n1 = 0.5;
                }
                if (den2 > 1e-12) {
                    n2 = qBound(0.0, qLn(qMax(r2, logMin2) / logMin2) / den2, 1.0);
                } else {
                    n2 = 0.5;
                }
            } else {
                n1 = qBound(0.0, (r1 - rho1Min) / rho1Span, 1.0);
                n2 = qBound(0.0, (r2 - rho2Min) / rho2Span, 1.0);
            }

            int R = static_cast<int>(255 * (c1R * n1 + c2R * n2 * (1.0 - n1)));
            int G = static_cast<int>(255 * (c1G * n1 + c2G * n2 * (1.0 - n1)));
            int B = static_cast<int>(255 * (c1B * n1 + c2B * n2 * (1.0 - n1)));

            pixels[dstIdx + 0] = static_cast<uchar>(qBound(0, R, 255));
            pixels[dstIdx + 1] = static_cast<uchar>(qBound(0, G, 255));
            pixels[dstIdx + 2] = static_cast<uchar>(qBound(0, B, 255));
            pixels[dstIdx + 3] = 255;
        }
    }

    if (texture_->isCreated()) {
        texture_->destroy();
    }

    texture_->create();
    texture_->setSize(nx, ny);
    texture_->setFormat(QOpenGLTexture::RGBA8_UNorm);
    texture_->setMinificationFilter(QOpenGLTexture::Linear);
    texture_->setMagnificationFilter(QOpenGLTexture::Nearest);
    texture_->setWrapMode(QOpenGLTexture::ClampToEdge);
    texture_->allocateStorage();
    texture_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, pixels.constData());
}

void HeatmapWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (!currentData_.isValid()) {
        return;
    }

    int nx = currentData_.meta.nx;
    int ny = currentData_.meta.ny;
    double Lx = currentData_.meta.Lx;
    double Ly = currentData_.meta.Ly;

    float dataAspect = Lx / Ly;
    float viewAspect = float(viewportWidth_) / float(viewportHeight_);
    float scaleX = 1.0f, scaleY = 1.0f, offsetX = 0.0f, offsetY = 0.0f;

    if (dataAspect > viewAspect) {
        scaleY = viewAspect / dataAspect;
        offsetY = (1.0f - scaleY) / 2.0f;
    } else {
        scaleX = dataAspect / viewAspect;
        offsetX = (1.0f - scaleX) / 2.0f;
    }

    float normX = float(event->pos().x()) / viewportWidth_;
    float normY = 1.0f - float(event->pos().y()) / viewportHeight_;

    if (normX < offsetX || normX > 1.0f - offsetX ||
        normY < offsetY || normY > 1.0f - offsetY) {
        return;
    }

    float dataX = (normX - offsetX) / scaleX;
    float dataY = (normY - offsetY) / scaleY;

    double x = dataX * Lx;
    double y = dataY * Ly;

    int ix = qBound(0, static_cast<int>(dataX * nx), nx - 1);
    int iy = qBound(0, static_cast<int>(dataY * ny), ny - 1);
    int idx = iy * nx + ix;

    double rho1 = currentData_.rho1[idx];
    double rho2 = currentData_.rho2[idx];

    emit cursorPosition(x, y, rho1, rho2);
}

} // namespace salr
