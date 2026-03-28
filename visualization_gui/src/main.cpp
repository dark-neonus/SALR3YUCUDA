/*
 * main.cpp - Application entry point for SALR Visualization GUI
 */

#include <QApplication>
#include <QDir>
#include <QCommandLineParser>
#include <QStyleFactory>

#include "MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Set application info
    app.setApplicationName("SALR Visualization");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("SALR");
    app.setOrganizationDomain("salr.local");

    // Use Fusion style for consistent look
    app.setStyle(QStyleFactory::create("Fusion"));

    // Command line parser
    QCommandLineParser parser;
    parser.setApplicationDescription("SALR DFT Visualization GUI");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption dbPathOption(
        QStringList() << "d" << "database",
        "Path to database directory",
        "path");
    parser.addOption(dbPathOption);

    parser.process(app);

    // Create and show main window
    salr::MainWindow window;

    // If database path was specified, use it
    if (parser.isSet(dbPathOption)) {
        QString dbPath = parser.value(dbPathOption);
        // Main window will handle this via settings or we could add a method
    }

    window.show();

    return app.exec();
}
