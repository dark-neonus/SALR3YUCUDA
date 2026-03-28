/*
 * ConfigEditorDialog.h - Full configuration editor dialog
 */

#ifndef SALR_GUI_CONFIG_EDITOR_DIALOG_H
#define SALR_GUI_CONFIG_EDITOR_DIALOG_H

#include <QDialog>
#include "Types.h"

class QTabWidget;
class QTextEdit;

namespace salr {

class ConfigEditorDialog : public QDialog {
    Q_OBJECT

public:
    explicit ConfigEditorDialog(QWidget* parent = nullptr);

    void setConfig(const SimulationConfig& config);
    SimulationConfig config() const;

private slots:
    void onLoadFile();
    void onSaveFile();
    void updatePreview();

private:
    void setupUi();

    QTabWidget* tabWidget_ = nullptr;
    QTextEdit* previewEdit_ = nullptr;

    // This would contain all the config widgets
    // For brevity, we just store the config directly
    SimulationConfig config_;
};

} // namespace salr

#endif // SALR_GUI_CONFIG_EDITOR_DIALOG_H
