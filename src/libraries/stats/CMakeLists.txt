# Define the library name
set(MNE_STATS_LIBRARY_NAME mne_stats)

set(QT_REQUIRED_COMPONENTS Core)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})

set(SOURCES
    stats_global.cpp
    ttest.cpp
    correction.cpp
)

set(HEADERS
    stats_global.h
    ttest.h
    correction.h
)

# Create the library
add_library(${MNE_STATS_LIBRARY_NAME} SHARED ${SOURCES} ${HEADERS})

set(QT_REQUIRED_COMPONENT_LIBS ${QT_REQUIRED_COMPONENTS})
list(TRANSFORM QT_REQUIRED_COMPONENT_LIBS PREPEND "Qt${QT_VERSION_MAJOR}::")

# Define the export macro
target_compile_definitions(${MNE_STATS_LIBRARY_NAME} PRIVATE MNE_STATS_LIBRARY)

# Link dependencies
target_link_libraries(${MNE_STATS_LIBRARY_NAME}
    ${QT_REQUIRED_COMPONENT_LIBS}
    mne_utils
    eigen
)

# Include directories
target_include_directories(${MNE_STATS_LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/libraries/stats>
)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${MNE_STATS_LIBRARY_NAME} PRIVATE STATICBUILD)
endif()

# Install rules
install(TARGETS ${MNE_STATS_LIBRARY_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS}
    DESTINATION include/libraries/stats
)
