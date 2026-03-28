/*
 * Types.h - Common type definitions for SALR Visualization GUI
 *
 * C++ structures mirroring the C database engine types for Qt integration.
 */

#ifndef SALR_GUI_TYPES_H
#define SALR_GUI_TYPES_H

#include <QString>
#include <QVector>
#include <QDateTime>
#include <memory>

namespace salr {

// Boundary condition modes (mirrors BoundaryMode in config.h)
enum class BoundaryMode {
    PBC,  // Periodic in both x and y
    W2,   // Hard walls at x=0 and x=Lx; periodic in y
    W4    // Hard walls on all four sides
};

inline QString boundaryModeToString(BoundaryMode mode) {
    switch (mode) {
        case BoundaryMode::PBC: return "PBC";
        case BoundaryMode::W2:  return "W2";
        case BoundaryMode::W4:  return "W4";
    }
    return "PBC";
}

inline BoundaryMode stringToBoundaryMode(const QString& str) {
    if (str == "W2") return BoundaryMode::W2;
    if (str == "W4") return BoundaryMode::W4;
    return BoundaryMode::PBC;
}

// Grid parameters
struct GridParams {
    int nx = 128;
    int ny = 128;
    double dx = 0.25;
    double dy = 0.25;
    double Lx = 32.0;  // Derived: dx * nx
    double Ly = 32.0;  // Derived: dy * ny

    void computeDerivedValues() {
        Lx = dx * nx;
        Ly = dy * ny;
    }
};

// Solver parameters
struct SolverParams {
    int maxIterations = 50000;
    double tolerance = 1.0e-8;
    double xi1 = 0.2;
    double xi2 = 0.2;
    double errorChangeThreshold = 1.0e-6;
    double xiDampingFactor = 0.5;
};

// Potential parameters (3-Yukawa)
struct PotentialParams {
    double cutoffRadius = 16.0;
    // A[i][j][m] - amplitudes, alpha[i][j][m] - decay rates
    // i,j in {0,1} for species, m in {0,1,2} for Yukawa terms
    double A[2][2][3] = {{{0}}};
    double alpha[2][2][3] = {{{0}}};
};

// Complete simulation configuration
struct SimulationConfig {
    GridParams grid;
    PotentialParams potential;
    SolverParams solver;
    BoundaryMode boundaryMode = BoundaryMode::PBC;
    double temperature = 2.9;
    double rho1 = 0.4;  // Bulk density species 1
    double rho2 = 0.2;  // Bulk density species 2
    QString outputDir = "output/";
    int saveEvery = 1000;
};

// Session information (mirrors RunSummary in registry.h)
struct SessionInfo {
    QString runId;
    QString nickname;
    QDateTime createdAt;
    double temperature = 0.0;
    double rho1Bulk = 0.0;
    double rho2Bulk = 0.0;
    int nx = 0;
    int ny = 0;
    double dx = 0.0;
    double dy = 0.0;
    QString boundaryMode;
    QString configHash;
    QString source;  // Source session ID if branched
    int snapshotCount = 0;
    double finalError = -1.0;
    bool converged = false;
};

// Snapshot metadata (mirrors SnapshotMeta in hdf5_io.h)
struct SnapshotMeta {
    int iteration = 0;
    double currentError = 0.0;
    double deltaError = 0.0;
    double temperature = 0.0;
    double rho1Bulk = 0.0;
    double rho2Bulk = 0.0;
    int nx = 0;
    int ny = 0;
    double Lx = 0.0;
    double Ly = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    QString boundaryMode;
    double xi1 = 0.0;
    double xi2 = 0.0;
    double cutoffRadius = 0.0;
    QDateTime createdAt;
};

// Snapshot data container (density arrays + metadata)
struct SnapshotData {
    QVector<double> rho1;  // Species 1 density (nx*ny)
    QVector<double> rho2;  // Species 2 density (nx*ny)
    SnapshotMeta meta;

    // Statistics
    double rho1Min = 0.0, rho1Max = 0.0, rho1Mean = 0.0;
    double rho2Min = 0.0, rho2Max = 0.0, rho2Mean = 0.0;

    void computeStatistics() {
        if (rho1.isEmpty() || rho2.isEmpty()) return;

        rho1Min = rho1Max = rho1[0];
        rho2Min = rho2Max = rho2[0];
        double sum1 = 0, sum2 = 0;

        for (int i = 0; i < rho1.size(); ++i) {
            double v1 = rho1[i], v2 = rho2[i];
            if (v1 < rho1Min) rho1Min = v1;
            if (v1 > rho1Max) rho1Max = v1;
            if (v2 < rho2Min) rho2Min = v2;
            if (v2 > rho2Max) rho2Max = v2;
            sum1 += v1;
            sum2 += v2;
        }

        rho1Mean = sum1 / rho1.size();
        rho2Mean = sum2 / rho2.size();
    }

    bool isValid() const {
        return !rho1.isEmpty() && !rho2.isEmpty() &&
               rho1.size() == meta.nx * meta.ny &&
               rho2.size() == meta.nx * meta.ny;
    }
};

// Session filter for queries
struct SessionFilter {
    double tempMin = 0.0;
    double tempMax = 0.0;
    double rho1Min = 0.0;
    double rho1Max = 0.0;
    bool convergedOnly = false;

    bool hasFilter() const {
        return tempMin > 0 || tempMax > 0 || rho1Min > 0 || rho1Max > 0 || convergedOnly;
    }
};

} // namespace salr

#endif // SALR_GUI_TYPES_H
