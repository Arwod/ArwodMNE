#==============================================================================================================
#
# @file     CMakeLists.txt
# @author   Kiro AI Assistant
# @since    0.1.0
# @date     January, 2025
#
# @section  LICENSE
#
# Copyright (C) 2025, MNE-CPP authors. All rights reserved.
#
# @brief    This project file builds the algorithms integration test example.
#
#==============================================================================================================

cmake_minimum_required(VERSION 3.14)
project(ex_algorithms_integration LANGUAGES CXX)

#Handle qt uic, moc, rcc automatically
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(SOURCES
    main.cpp
    ../../libraries/integration.cpp
    ../../libraries/benchmark.cpp
)

set(QT_REQUIRED_COMPONENTS Core)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})

add_executable(${PROJECT_NAME} ${SOURCES})

set(QT_REQUIRED_COMPONENT_LIBS ${QT_REQUIRED_COMPONENTS})
list(TRANSFORM QT_REQUIRED_COMPONENT_LIBS PREPEND "Qt${QT_VERSION_MAJOR}::")

set(MNE_LIBS_REQUIRED
    mne_utils
    mne_fiff
    mne_mne
    mne_connectivity
    mne_dataio
    mne_channels
    mne_rtprocessing
    mne_decoding
    mne_stats
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${QT_REQUIRED_COMPONENT_LIBS}
    ${MNE_LIBS_REQUIRED}
)

target_include_directories(${PROJECT_NAME} PRIVATE ../../libraries)

target_compile_definitions(${PROJECT_NAME} PRIVATE MNE_GIT_HASH_SHORT="${MNE_GIT_HASH_SHORT}" MNE_GIT_HASH_LONG="${MNE_GIT_HASH_LONG}")

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE STATICBUILD)
endif()
