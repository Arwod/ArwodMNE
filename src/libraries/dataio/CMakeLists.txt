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
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that
# the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
#       following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#       the following disclaimer in the documentation and/or other materials provided with the distribution.
#     * Neither the name of the Massachusetts General Hospital nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MASSACHUSETTS GENERAL HOSPITAL BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
# @brief    This project file builds the data I/O library.
#
#==============================================================================================================

cmake_minimum_required(VERSION 3.14)
project(mne_dataio LANGUAGES CXX)

#Handle qt uic, moc, rcc automatically
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(SOURCES
    dataio_global.cpp
    multiformat_reader.cpp
    data_converter.cpp
)

set(HEADERS
    dataio_global.h
    multiformat_reader.h
    data_converter.h
)

set(FILE_TO_UPDATE dataio_global.cpp)

set(SOURCE_PATHS ${SOURCES})
list(TRANSFORM SOURCE_PATHS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set_source_files_properties(${FILE_TO_UPDATE} PROPERTIES OBJECT_DEPENDS "${SOURCE_PATHS}")

set(QT_REQUIRED_COMPONENTS Core)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set(FFTW_LIBS "")

target_include_directories(${PROJECT_NAME} PUBLIC ../)

if(USE_FFTW)
    if(WIN32)
        set(FFTW_LIBS
            ${FFTW_DIR_LIBS}/libfftw3-3.dll
            ${FFTW_DIR_LIBS}/libfftw3f-3.dll
            ${FFTW_DIR_LIBS}/libfftw3l-3.dll
        )
        target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW_DIR_INCLUDE})
    elseif(UNIX AND NOT APPLE)
        set(FFTW_LIBS ${FFTW_DIR_LIBS}/lib/libfftw3.a ${FFTW_DIR_LIBS}/lib/libfftw3_threads.a)
        target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW_DIR_INCLUDE}/include)
    endif()
endif()

set(QT_REQUIRED_COMPONENT_LIBS ${QT_REQUIRED_COMPONENTS})

list(TRANSFORM QT_REQUIRED_COMPONENT_LIBS PREPEND "Qt${QT_VERSION_MAJOR}::")

set(MNE_LIBS_REQUIRED
    mne_utils
    mne_fiff
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${QT_REQUIRED_COMPONENT_LIBS}
    ${MNE_LIBS_REQUIRED}
    ${FFTW_LIBS}
    eigen)

target_compile_definitions(${PROJECT_NAME} PRIVATE DATAIO_LIBRARY MNE_GIT_HASH_SHORT="${MNE_GIT_HASH_SHORT}" MNE_GIT_HASH="${MNE_GIT_HASH}")

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE STATICBUILD)
endif()
