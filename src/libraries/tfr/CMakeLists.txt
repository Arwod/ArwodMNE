cmake_minimum_required(VERSION 3.14)
project(mne_tfr LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(QT_REQUIRED_COMPONENTS Core Concurrent)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})

set(SOURCES
    tfr_global.cpp
    tfr_utils.cpp
    tfr_compute.cpp
    timefrequency.cpp
    psd.cpp
    csd.cpp
)

set(HEADERS
    tfr_global.h
    tfr_utils.h
    tfr_compute.h
    timefrequency.h
    psd.h
    csd.h
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set(QT_REQUIRED_COMPONENT_LIBS ${QT_REQUIRED_COMPONENTS})
list(TRANSFORM QT_REQUIRED_COMPONENT_LIBS PREPEND "Qt${QT_VERSION_MAJOR}::")

target_link_libraries(${PROJECT_NAME}
    ${QT_REQUIRED_COMPONENT_LIBS}
    mne_utils
    eigen
)

target_compile_definitions(${PROJECT_NAME} PRIVATE MNE_TFR_LIBRARY)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/libraries/tfr>
)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE STATICBUILD)
endif()

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS}
    DESTINATION include/libraries/tfr
)
