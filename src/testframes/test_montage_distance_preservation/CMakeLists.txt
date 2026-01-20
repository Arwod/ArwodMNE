cmake_minimum_required(VERSION 3.16)

project(test_montage_distance_preservation LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Core Test REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(test_montage_distance_preservation
    test_montage_distance_preservation.cpp
)

target_link_libraries(test_montage_distance_preservation
    Qt6::Core
    Qt6::Test
    Eigen3::Eigen
    mne_fiff
    mne_utils
)

target_include_directories(test_montage_distance_preservation PRIVATE
    ${CMAKE_SOURCE_DIR}/src/libraries
)
