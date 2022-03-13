# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.5)
set(CMAKE_CXX_FLAGS   "-std=c++17")             # c++17

project(bio)

aux_source_directory(../bio DIR_SRCS)
aux_source_directory(../utils DIR_SRCS)
add_library (tool ${DIR_SRCS})
file(GLOB DIR_HEADERS ../bio/*.h ../utils/*.h)

set_target_properties(tool PROPERTIES PUBLIC_HEADER "${DIR_HEADERS}")

SET(CMAKE_INSTALL_PREFIX ".")
INSTALL(TARGETS tool
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include
)