cmake_minimum_required(VERSION 3.16)

project(test_data_compatibility LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Core Test REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(test_data_compatibility
    test_data_compatibility.cpp
)

target_link_libraries(test_data_compatibility
    Qt6::Core
    Qt6::Test
    Eigen3::Eigen
    mne_fiff
    mne_mne
    mne_utils
)

target_include_directories(test_data_compatibility PRIVATE
    ${CMAKE_SOURCE_DIR}/src/libraries
)
