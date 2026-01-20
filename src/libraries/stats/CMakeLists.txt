# Set the library name
set(MNE_LIBRARY_NAME mne_stats)

set(QT_REQUIRED_COMPONENTS Core Concurrent)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})

if(QT_VERSION_MAJOR EQUAL 6)
    find_package(Qt6 REQUIRED COMPONENTS Core5Compat)
endif()

# Define the source files
set(MNE_LIBRARY_SOURCES
    stats_global.cpp
    statistics.cpp
    clusterpermutation.cpp
    correction.cpp
    ttest.cpp
)

# Define the header files
set(MNE_LIBRARY_HEADERS
    stats_global.h
    statistics.h
    clusterpermutation.h
    correction.h
    ttest.h
)

# Create the library
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_library(${MNE_LIBRARY_NAME} STATIC ${MNE_LIBRARY_SOURCES} ${MNE_LIBRARY_HEADERS})
else()
    add_library(${MNE_LIBRARY_NAME} STATIC ${MNE_LIBRARY_SOURCES} ${MNE_LIBRARY_HEADERS})
endif()

# Define definitions
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${MNE_LIBRARY_NAME} PRIVATE STATICBUILD)
endif()

# Define include directories
target_include_directories(${MNE_LIBRARY_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Define dependencies
target_link_libraries(${MNE_LIBRARY_NAME} PUBLIC
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Concurrent
    eigen
)

if(QT_VERSION_MAJOR EQUAL 6)
    target_link_libraries(${MNE_LIBRARY_NAME} PUBLIC Qt6::Core5Compat)
endif()

# Set properties
set_target_properties(${MNE_LIBRARY_NAME} PROPERTIES
    VERSION ${MNE_LIBRARY_VERSION}
    SOVERSION ${MNE_LIBRARY_SOVERSION}
)

# Install
install(TARGETS ${MNE_LIBRARY_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${MNE_LIBRARY_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stats
)
