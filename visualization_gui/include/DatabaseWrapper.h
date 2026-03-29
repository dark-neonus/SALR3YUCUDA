/*
 * DatabaseWrapper.h - C++ wrapper for the SALR database engine
 *
 * Provides a Qt-friendly interface to the C database API.
 */

#ifndef SALR_GUI_DATABASE_WRAPPER_H
#define SALR_GUI_DATABASE_WRAPPER_H

#include <QObject>
#include <QString>
#include <QVector>
#include <QMutex>
#include <QRecursiveMutex>
#include "Types.h"

// Forward declarations for C types
struct DbRun;
struct SimConfig;

namespace salr {

class DatabaseWrapper : public QObject {
    Q_OBJECT

public:
    explicit DatabaseWrapper(QObject* parent = nullptr);
    ~DatabaseWrapper();

    // Initialization
    bool initialize(const QString& dataRoot);
    void close();
    bool isInitialized() const { return initialized_; }
    QString dataRoot() const { return dataRoot_; }

    // Session operations
    QVector<SessionInfo> listSessions(const SessionFilter& filter = SessionFilter());
    SessionInfo getSessionInfo(const QString& runId);
    bool setNickname(const QString& runId, const QString& nickname);
    bool deleteSession(const QString& runId);
    bool sessionExists(const QString& runId);

    // Snapshot operations
    QVector<int> listSnapshots(const QString& runId);
    SnapshotData loadSnapshot(const QString& runId, int iteration = -1);
    SnapshotMeta loadMetadata(const QString& runId, int iteration = -1);

    // Config operations
    bool loadConfigFile(const QString& path, SimulationConfig& config);
    bool saveConfigFile(const QString& path, const SimulationConfig& config);
    SimulationConfig configFromSession(const SessionInfo& session);

    // Path utilities
    QString sessionPath(const QString& runId) const;
    QString snapshotPath(const QString& runId, int iteration) const;
    QString executablePath(bool useCuda) const;

signals:
    void sessionListChanged();
    void errorOccurred(const QString& message);

private:
    QString dataRoot_;
    bool initialized_ = false;
    mutable QRecursiveMutex mutex_;

    // Helper to convert C RunSummary to SessionInfo
    SessionInfo convertRunSummary(const void* summary);

    // Helper to convert C SnapshotMeta to our SnapshotMeta
    SnapshotMeta convertSnapshotMeta(const void* meta);
};

} // namespace salr

#endif // SALR_GUI_DATABASE_WRAPPER_H
