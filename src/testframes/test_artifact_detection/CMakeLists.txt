#=============================================================================================================
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
# @brief    CMake configuration for artifact detection unit test
#
#=============================================================================================================

cmake_minimum_required(VERSION 3.14)

project(test_artifact_detection LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Test)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Test)
find_package(Eigen3 REQUIRED)

set(SOURCES
    test_artifact_detection.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Test
    Eigen3::Eigen
    mne_utils
    mne_preprocessing
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../libraries
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../out/Release/tests
)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
