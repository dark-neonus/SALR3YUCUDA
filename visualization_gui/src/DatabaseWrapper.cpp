/*
 * DatabaseWrapper.cpp - Implementation of the database wrapper
 */

#include "DatabaseWrapper.h"
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QRegularExpression>
#include <QCoreApplication>
#include <cstring>

// Include C database engine headers
extern "C" {
#include "db_engine.h"
#include "registry.h"
#include "hdf5_io.h"
}

// Include config.h for SimConfig structure
extern "C" {
#include "config.h"
}

namespace salr {

DatabaseWrapper::DatabaseWrapper(QObject* parent)
    : QObject(parent)
{
}

DatabaseWrapper::~DatabaseWrapper()
{
    close();
}

bool DatabaseWrapper::initialize(const QString& dataRoot)
{
    QMutexLocker locker(&mutex_);

    if (initialized_) {
        close();
    }

    dataRoot_ = dataRoot;
    QByteArray pathBytes = dataRoot.toUtf8();

    DbError err = db_init(pathBytes.constData());
    if (err != DB_OK) {
        emit errorOccurred(tr("Failed to initialize database at %1").arg(dataRoot));
        return false;
    }

    initialized_ = true;
    return true;
}

void DatabaseWrapper::close()
{
    QMutexLocker locker(&mutex_);

    if (initialized_) {
        db_close();
        initialized_ = false;
    }
}

QVector<SessionInfo> DatabaseWrapper::listSessions(const SessionFilter& filter)
{
    QMutexLocker locker(&mutex_);
    QVector<SessionInfo> result;

    if (!initialized_) {
        emit errorOccurred(tr("Database not initialized"));
        return result;
    }

    RunSummary* runs = nullptr;
    int count = 0;

    DbError err = db_registry_list(
        filter.tempMin, filter.tempMax,
        filter.rho1Min, filter.rho1Max,
        &runs, &count
    );

    if (err != DB_OK) {
        emit errorOccurred(tr("Failed to list sessions"));
        return result;
    }

    result.reserve(count);
    for (int i = 0; i < count; ++i) {
        SessionInfo info = convertRunSummary(&runs[i]);
        if (!filter.convergedOnly || info.converged) {
            result.append(info);
        }
    }

    free(runs);
    return result;
}

SessionInfo DatabaseWrapper::getSessionInfo(const QString& runId)
{
    QMutexLocker locker(&mutex_);
    SessionInfo info;

    if (!initialized_) {
        return info;
    }

    RunSummary summary;
    QByteArray idBytes = runId.toUtf8();

    RegistryError err = registry_get_run(idBytes.constData(), &summary);
    if (err == REG_OK) {
        info = convertRunSummary(&summary);
    }

    return info;
}

bool DatabaseWrapper::setNickname(const QString& runId, const QString& nickname)
{
    QMutexLocker locker(&mutex_);

    if (!initialized_) {
        return false;
    }

    QByteArray idBytes = runId.toUtf8();
    QByteArray nickBytes = nickname.toUtf8();

    DbError err = db_registry_set_nickname(
        idBytes.constData(),
        nickname.isEmpty() ? nullptr : nickBytes.constData()
    );

    if (err == DB_OK) {
        emit sessionListChanged();
        return true;
    }

    emit errorOccurred(tr("Failed to set nickname for session %1").arg(runId));
    return false;
}

bool DatabaseWrapper::deleteSession(const QString& runId)
{
    QMutexLocker locker(&mutex_);

    if (!initialized_) {
        return false;
    }

    QByteArray idBytes = runId.toUtf8();
    DbError err = db_registry_delete_run(idBytes.constData());

    if (err == DB_OK) {
        emit sessionListChanged();
        return true;
    }

    emit errorOccurred(tr("Failed to delete session %1").arg(runId));
    return false;
}

bool DatabaseWrapper::sessionExists(const QString& runId)
{
    QMutexLocker locker(&mutex_);

    if (!initialized_) {
        return false;
    }

    QByteArray idBytes = runId.toUtf8();
    return registry_run_exists(idBytes.constData()) != 0;
}

QVector<int> DatabaseWrapper::listSnapshots(const QString& runId)
{
    QMutexLocker locker(&mutex_);
    QVector<int> result;

    if (!initialized_) {
        return result;
    }

    QByteArray idBytes = runId.toUtf8();

    DbRun* run = nullptr;
    DbError err = db_run_open(idBytes.constData(), &run);
    if (err != DB_OK) {
        return result;
    }

    int* iters = nullptr;
    int count = 0;

    err = db_snapshot_list(run, &iters, &count);
    if (err == DB_OK && iters != nullptr) {
        result.reserve(count);
        for (int i = 0; i < count; ++i) {
            result.append(iters[i]);
        }
        free(iters);
    }

    db_run_close(run);
    return result;
}

SnapshotData DatabaseWrapper::loadSnapshot(const QString& runId, int iteration)
{
    QMutexLocker locker(&mutex_);
    SnapshotData data;

    if (!initialized_) {
        emit errorOccurred(tr("Database not initialized"));
        return data;
    }

    QByteArray idBytes = runId.toUtf8();

    DbRun* run = nullptr;
    DbError err = db_run_open(idBytes.constData(), &run);
    if (err != DB_OK) {
        emit errorOccurred(tr("Failed to open session %1").arg(runId));
        return data;
    }

    // If iteration is -1, find the latest snapshot
    if (iteration < 0) {
        int* iters = nullptr;
        int count = 0;
        if (db_snapshot_list(run, &iters, &count) == DB_OK && count > 0) {
            iteration = iters[count - 1];  // Last (highest) iteration
            free(iters);
        } else {
            db_run_close(run);
            emit errorOccurred(tr("No snapshots found in session %1").arg(runId));
            return data;
        }
    }

    // Build snapshot path and read metadata
    QString snapPath = snapshotPath(runId, iteration);
    QByteArray pathBytes = snapPath.toUtf8();

    ::SnapshotMeta cMeta;
    memset(&cMeta, 0, sizeof(cMeta));
    HDF5Error hErr = hdf5_read_metadata(pathBytes.constData(), &cMeta);
    if (hErr != HDF5_OK) {
        db_run_close(run);
        emit errorOccurred(tr("Failed to read snapshot metadata"));
        return data;
    }

    data.meta = convertSnapshotMeta(&cMeta);

    // Allocate and load density arrays
    int size = cMeta.nx * cMeta.ny;
    data.rho1.resize(size);
    data.rho2.resize(size);

    err = db_snapshot_load(run, iteration,
                           data.rho1.data(), data.rho2.data(),
                           nullptr);

    db_run_close(run);

    if (err != DB_OK) {
        data.rho1.clear();
        data.rho2.clear();
        emit errorOccurred(tr("Failed to load snapshot data"));
        return data;
    }

    data.computeStatistics();
    return data;
}

SnapshotMeta DatabaseWrapper::loadMetadata(const QString& runId, int iteration)
{
    QMutexLocker locker(&mutex_);
    SnapshotMeta result;

    if (!initialized_) {
        return result;
    }

    // If iteration is -1, find the latest
    if (iteration < 0) {
        QVector<int> snaps = listSnapshots(runId);
        if (snaps.isEmpty()) {
            return result;
        }
        iteration = snaps.last();
    }

    QString snapPath = snapshotPath(runId, iteration);
    QByteArray pathBytes = snapPath.toUtf8();

    ::SnapshotMeta cMeta;
    HDF5Error err = hdf5_read_metadata(pathBytes.constData(), &cMeta);
    if (err == HDF5_OK) {
        result = convertSnapshotMeta(&cMeta);
    }

    return result;
}

bool DatabaseWrapper::loadConfigFile(const QString& path, SimulationConfig& config)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        emit errorOccurred(tr("Cannot open config file: %1").arg(path));
        return false;
    }

    QString currentSection;
    QTextStream in(&file);

    // Set defaults
    config = SimulationConfig();

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();

        // Skip empty lines and comments
        if (line.isEmpty() || line.startsWith('#')) {
            continue;
        }

        // Section header
        if (line.startsWith('[') && line.endsWith(']')) {
            currentSection = line.mid(1, line.length() - 2).toLower();
            continue;
        }

        // Key = value
        int eqPos = line.indexOf('=');
        if (eqPos < 0) continue;

        QString key = line.left(eqPos).trimmed().toLower();
        QString value = line.mid(eqPos + 1).trimmed();

        // Remove inline comments
        int commentPos = value.indexOf('#');
        if (commentPos >= 0) {
            value = value.left(commentPos).trimmed();
        }

        // Parse based on section
        if (currentSection == "grid") {
            if (key == "nx") config.grid.nx = value.toInt();
            else if (key == "ny") config.grid.ny = value.toInt();
            else if (key == "dx") config.grid.dx = value.toDouble();
            else if (key == "dy") config.grid.dy = value.toDouble();
            else if (key == "boundary_mode") {
                config.boundaryMode = stringToBoundaryMode(value.toUpper());
            }
        }
        else if (currentSection == "physics") {
            if (key == "temperature") config.temperature = value.toDouble();
            else if (key == "rho1") config.rho1 = value.toDouble();
            else if (key == "rho2") config.rho2 = value.toDouble();
            else if (key == "cutoff_radius") config.potential.cutoffRadius = value.toDouble();
        }
        else if (currentSection == "interaction") {
            // Parse A_IJ_M and a_IJ_M
            QRegularExpression re("^([Aa])_(\\d)(\\d)_(\\d)$");
            QRegularExpressionMatch match = re.match(key);
            if (match.hasMatch()) {
                bool isA = match.captured(1).toUpper() == "A";
                int i = match.captured(2).toInt() - 1;
                int j = match.captured(3).toInt() - 1;
                int m = match.captured(4).toInt() - 1;

                if (i >= 0 && i < 2 && j >= 0 && j < 2 && m >= 0 && m < 3) {
                    double v = value.toDouble();
                    if (isA) {
                        config.potential.A[i][j][m] = v;
                        config.potential.A[j][i][m] = v;  // Symmetric
                    } else {
                        config.potential.alpha[i][j][m] = v;
                        config.potential.alpha[j][i][m] = v;
                    }
                }
            }
        }
        else if (currentSection == "solver") {
            if (key == "max_iterations") config.solver.maxIterations = value.toInt();
            else if (key == "tolerance") config.solver.tolerance = value.toDouble();
            else if (key == "xi1") config.solver.xi1 = value.toDouble();
            else if (key == "xi2") config.solver.xi2 = value.toDouble();
            else if (key == "error_change_threshold") config.solver.errorChangeThreshold = value.toDouble();
            else if (key == "xi_damping_factor") config.solver.xiDampingFactor = value.toDouble();
        }
        else if (currentSection == "output") {
            if (key == "output_dir") config.outputDir = value;
            else if (key == "save_every") config.saveEvery = value.toInt();
        }
    }

    config.grid.computeDerivedValues();
    return true;
}

bool DatabaseWrapper::saveConfigFile(const QString& path, const SimulationConfig& config)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        emit errorOccurred(tr("Cannot write config file: %1").arg(path));
        return false;
    }

    QTextStream out(&file);
    out.setRealNumberPrecision(10);

    // Grid section
    out << "[grid]\n";
    out << "dx = " << config.grid.dx << "\n";
    out << "dy = " << config.grid.dy << "\n";
    out << "nx = " << config.grid.nx << "\n";
    out << "ny = " << config.grid.ny << "\n";
    out << "boundary_mode = " << boundaryModeToString(config.boundaryMode) << "\n";
    out << "\n";

    // Physics section
    out << "[physics]\n";
    out << "temperature = " << config.temperature << "\n";
    out << "rho1 = " << config.rho1 << "\n";
    out << "rho2 = " << config.rho2 << "\n";
    out << "cutoff_radius = " << config.potential.cutoffRadius << "\n";
    out << "\n";

    // Interaction section
    out << "[interaction]\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = i; j < 2; ++j) {
            for (int m = 0; m < 3; ++m) {
                out << QString("A_%1%2_%3 = ").arg(i+1).arg(j+1).arg(m+1)
                    << config.potential.A[i][j][m] << "\n";
            }
            for (int m = 0; m < 3; ++m) {
                out << QString("a_%1%2_%3 = ").arg(i+1).arg(j+1).arg(m+1)
                    << config.potential.alpha[i][j][m] << "\n";
            }
            out << "\n";
        }
    }

    // Solver section
    out << "[solver]\n";
    out << "max_iterations = " << config.solver.maxIterations << "\n";
    out << "tolerance = " << config.solver.tolerance << "\n";
    out << "xi1 = " << config.solver.xi1 << "\n";
    out << "xi2 = " << config.solver.xi2 << "\n";
    out << "error_change_threshold = " << config.solver.errorChangeThreshold << "\n";
    out << "xi_damping_factor = " << config.solver.xiDampingFactor << "\n";
    out << "\n";

    // Output section
    out << "[output]\n";
    out << "output_dir = " << config.outputDir << "\n";
    out << "save_every = " << config.saveEvery << "\n";

    return true;
}

SimulationConfig DatabaseWrapper::configFromSession(const SessionInfo& session)
{
    SimulationConfig config;

    config.grid.nx = session.nx;
    config.grid.ny = session.ny;
    config.grid.dx = session.dx;
    config.grid.dy = session.dy;
    config.grid.computeDerivedValues();

    config.boundaryMode = stringToBoundaryMode(session.boundaryMode);
    config.temperature = session.temperature;
    config.rho1 = session.rho1Bulk;
    config.rho2 = session.rho2Bulk;

    // Note: We don't have potential/solver params in SessionInfo
    // These would need to be loaded from a snapshot if needed

    return config;
}

QString DatabaseWrapper::sessionPath(const QString& runId) const
{
    return QDir(dataRoot_).filePath(runId);
}

QString DatabaseWrapper::snapshotPath(const QString& runId, int iteration) const
{
    QString filename = QString("snapshot_%1.h5").arg(iteration, 6, 10, QChar('0'));
    return QDir(sessionPath(runId)).filePath(filename);
}

QString DatabaseWrapper::executablePath(bool useCuda) const
{
    // Find executable relative to application directory
    QDir appDir(QCoreApplication::applicationDirPath());

    QString exeName = useCuda ? "salr_dft_cuda_db" : "salr_dft_db";
#ifdef Q_OS_WIN
    exeName += ".exe";
#endif

    // Try current directory first, then parent directories
    QStringList searchPaths = {
        appDir.filePath(exeName),
        appDir.filePath("../" + exeName),
        appDir.filePath("../build/" + exeName)
    };

    for (const QString& path : searchPaths) {
        if (QFile::exists(path)) {
            return QDir::cleanPath(path);
        }
    }

    return exeName;  // Return just the name, hope it's in PATH
}

SessionInfo DatabaseWrapper::convertRunSummary(const void* ptr)
{
    const RunSummary* summary = static_cast<const RunSummary*>(ptr);
    SessionInfo info;

    info.runId = QString::fromUtf8(summary->run_id);
    info.nickname = QString::fromUtf8(summary->nickname);
    info.createdAt = QDateTime::fromString(
        QString::fromUtf8(summary->created_at), Qt::ISODate);
    info.temperature = summary->temperature;
    info.rho1Bulk = summary->rho1_bulk;
    info.rho2Bulk = summary->rho2_bulk;
    info.nx = summary->nx;
    info.ny = summary->ny;
    info.dx = summary->dx;
    info.dy = summary->dy;
    info.boundaryMode = QString::fromUtf8(summary->boundary_mode);
    info.configHash = QString::fromUtf8(summary->config_hash);
    info.source = QString::fromUtf8(summary->source);
    info.snapshotCount = summary->snapshot_count;
    info.finalError = summary->final_error;
    info.converged = (summary->converged != 0);

    return info;
}

SnapshotMeta DatabaseWrapper::convertSnapshotMeta(const void* ptr)
{
    const ::SnapshotMeta* meta = static_cast<const ::SnapshotMeta*>(ptr);
    SnapshotMeta result;

    result.iteration = meta->iteration;
    result.currentError = meta->current_error;
    result.deltaError = meta->delta_error;
    result.temperature = meta->temperature;
    result.rho1Bulk = meta->rho1_bulk;
    result.rho2Bulk = meta->rho2_bulk;
    result.nx = meta->nx;
    result.ny = meta->ny;
    result.Lx = meta->Lx;
    result.Ly = meta->Ly;
    result.dx = meta->dx;
    result.dy = meta->dy;
    result.boundaryMode = QString::fromUtf8(meta->boundary_mode);
    result.xi1 = meta->xi1;
    result.xi2 = meta->xi2;
    result.cutoffRadius = meta->cutoff_radius;
    result.createdAt = QDateTime::fromString(
        QString::fromUtf8(meta->created_at), Qt::ISODate);

    return result;
}

} // namespace salr
