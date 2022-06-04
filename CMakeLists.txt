
project(Stag)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -w  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -w ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
file(GLOB SRC_FILE1 "src/*.c*")
file(GLOB SRC_FILE2 "src/ED/*.c*")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    src/
    src/ED/
)
add_library(
    LibStag
    ${SRC_FILE1}
    ${SRC_FILE2}
)
target_link_libraries(
    LibStag
    ${OpenCV_LIBS}
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/app)
set_property(TARGET LibStag PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_executable(testrun
    main.cpp
)
# set_property(TARGET testrun PROPERTY
#   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_link_libraries(testrun
    LibStag
)