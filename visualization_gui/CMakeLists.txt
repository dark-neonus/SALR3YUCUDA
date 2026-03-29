# visualization_gui/CMakeLists.txt
# Qt-based visualization GUI for SALR DFT simulations

cmake_minimum_required(VERSION 3.18)
project(SALRVisualizationGUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt settings
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6 (required)
find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets Concurrent)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Source files
set(GUI_SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/DatabaseWrapper.cpp
    src/SessionBrowserWidget.cpp
    src/SnapshotBrowserWidget.cpp
    src/VisualizationWidget.cpp
    src/ScatterPlotWidget.cpp
    src/HeatmapWidget.cpp
    src/ParameterDisplayWidget.cpp
    src/RunControlWidget.cpp
    src/ConfigEditorDialog.cpp
    src/SnapshotSelectorDialog.cpp
    src/SimulationRunner.cpp
)

set(GUI_HEADERS
    include/Types.h
    include/MainWindow.h
    include/DatabaseWrapper.h
    include/SessionBrowserWidget.h
    include/SnapshotBrowserWidget.h
    include/VisualizationWidget.h
    include/ScatterPlotWidget.h
    include/HeatmapWidget.h
    include/ParameterDisplayWidget.h
    include/RunControlWidget.h
    include/ConfigEditorDialog.h
    include/SnapshotSelectorDialog.h
    include/SimulationRunner.h
)

# Create executable
add_executable(salr_gui ${GUI_SOURCES} ${GUI_HEADERS})

# Include directories
target_include_directories(salr_gui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/database_engine/include
    ${HDF5_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(salr_gui PRIVATE
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    Qt6::Concurrent
    OpenGL::GL
    salr_db_engine
    ${HDF5_LIBRARIES}
)

# Copy shaders to build directory
file(GLOB SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")
if(SHADER_FILES)
    file(COPY ${SHADER_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)
endif()

# Windows-specific: deploy Qt DLLs
if(WIN32)
    # Find windeployqt
    get_target_property(QMAKE_EXE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QMAKE_EXE}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")

    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET salr_gui POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                    --no-translations
                    --no-system-d3d-compiler
                    --no-opengl-sw
                    "$<TARGET_FILE:salr_gui>"
            COMMENT "Running windeployqt..."
        )
    endif()
endif()

message(STATUS "SALR Visualization GUI configured")
