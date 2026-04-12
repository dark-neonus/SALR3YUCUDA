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

    app.setApplicationName("SALR Visualization");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("SALR");
    app.setOrganizationDomain("salr.local");

    app.setStyle(QStyleFactory::create("Fusion"));

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

    salr::MainWindow window;

    window.show();

    return app.exec();
}
