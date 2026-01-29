set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_OUTPUT_DIRECTORY}/tests)

if(WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_OUTPUT_DIRECTORY}/apps)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
  if(NOT BUILD_MAC_APP_BUNDLE)
    set(CMAKE_MACOSX_RPATH TRUE)
  endif()
else()
  set(CMAKE_INSTALL_RPATH "\${ORIGIN}/../lib")
endif()

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)





# Restructured subdirectories
add_subdirectory(channels)
add_subdirectory(connectivity)
add_subdirectory(decoding)
add_subdirectory(fiff)
add_subdirectory(fwd)
add_subdirectory(inverse)
add_subdirectory(mne)
add_subdirectory(preprocessing)
add_subdirectory(rtprocessing)
add_subdirectory(simulation)
add_subdirectory(stats)
add_subdirectory(tfr)
add_subdirectory(utils)
