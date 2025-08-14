cmake_minimum_required(VERSION 3.15)
project(HelpCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译类型为Release
set(CMAKE_BUILD_TYPE Release)

# 针对AMD CPU的优化选项
if(MSVC)
    # Visual Studio编译器
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /Gy /fp:fast")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
else()
    # MinGW/GCC编译器
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-static -static-libgcc -static-libstdc++")
endif()

# 定义导出宏
add_definitions(-DHELPCPP_EXPORTS)

# 创建共享库
add_library(help_cpp SHARED
    help_cpp.cpp
    help_cpp.h
)

# 设置输出名称
set_target_properties(help_cpp PROPERTIES
    OUTPUT_NAME "help_cpp"
    PREFIX ""
    SUFFIX ".dll"
)

# 包含目录
target_include_directories(help_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 链接库
if(WIN32)
    target_link_libraries(help_cpp)
endif()