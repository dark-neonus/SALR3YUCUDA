/*
 * main.cpp - Application entry point for SALR Visualization GUI
 */

#include <QApplication>
#include <QDir>
#include <QCommandLineParser>
#include <QStyleFactory>
#include <QSize>
#include <QTimer>

#include "MainWindow.h"
#include "HeadlessController.h"

int main(int argc, char *argv[])
{
    // 1. Check for headless mode in raw args to set attributes
    bool headlessRequested = false;
    for (int i = 1; i < argc; ++i) {
        if (QString::fromLocal8Bit(argv[i]) == "--headless") {
            headlessRequested = true;
            break;
        }
    }

    if (headlessRequested) {
        // Essential for headless environments (like Docker or SSH)
        QCoreApplication::setAttribute(Qt::AA_UseSoftwareOpenGL);
        QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    }

    // 2. ALWAYS use QApplication because your Controller creates QWidgets.
    // The Python script passes "-platform offscreen", which tells QApplication 
    // not to look for a monitor.
    QApplication app(argc, argv);

    app.setApplicationName("SALR Visualization");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("SALR");
    app.setOrganizationDomain("salr.local");

    // Fusion style only works/matters if we have a GUI
    if (!headlessRequested) {
        app.setStyle(QStyleFactory::create("Fusion"));
    }

    QCommandLineParser parser;
    parser.setApplicationDescription("SALR DFT Visualization GUI");
    parser.addHelpOption();
    parser.addVersionOption();

    // Define Options
    QCommandLineOption dbPathOption({"d", "database"}, "Path to database", "path");
    QCommandLineOption headlessOption("headless", "Run in headless mode");
    QCommandLineOption configOption({"c", "config"}, "Config file", "path");
    QCommandLineOption backendOption({"b", "backend"}, "cpu/cuda", "backend", "cpu");
    QCommandLineOption sessionOption({"s", "session"}, "run_id", "run_id");
    QCommandLineOption widthOption("width", "width", "px", "1200");
    QCommandLineOption heightOption("height", "height", "px", "900");

    parser.addOption(dbPathOption);
    parser.addOption(headlessOption);
    parser.addOption(configOption);
    parser.addOption(backendOption);
    parser.addOption(sessionOption);
    parser.addOption(widthOption);
    parser.addOption(heightOption);

    parser.process(app);

    QString databasePath = parser.value(dbPathOption).trimmed();

    if (parser.isSet(headlessOption)) {
        // 3. Headless Lifecycle
        salr::HeadlessController::Options options;
        options.databasePath = databasePath;
        options.configPath = parser.value(configOption).trimmed();
        options.backend = parser.value(backendOption).trimmed();
        options.sessionId = parser.value(sessionOption).trimmed();
        options.renderSize = QSize(parser.value(widthOption).toInt(), parser.value(heightOption).toInt());

        // Create the controller. Pass &app as parent so it cleans up.
        auto* controller = new salr::HeadlessController(options, &app);
        
        // Start after event loop begins
        QTimer::singleShot(0, controller, &salr::HeadlessController::start);
        
        return app.exec();
    }

    // 4. GUI Lifecycle
    salr::MainWindow window(databasePath);
    window.show();

    return app.exec();
}