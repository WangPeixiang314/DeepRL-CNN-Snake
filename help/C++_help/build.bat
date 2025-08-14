@echo off
echo Building help_cpp.dll for Windows...

REM 检查g++是否存在
where g++ >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: g++ not found. Please install MinGW-w64 or use MSVC.
    pause
    exit /b 1
)

REM 设置编译参数
set CXX=g++
set CXXFLAGS=-O3 -march=native -mtune=native -ffast-math -fPIC -std=c++17 -DHELPCPP_EXPORTS
set LDFLAGS=-static -static-libgcc -static-libstdc++ -shared

REM 编译DLL
echo Compiling...
%CXX% %CXXFLAGS% %LDFLAGS% -o help_cpp.dll help_cpp.cpp

if %errorlevel% neq 0 (
    echo Error: Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful! DLL created: help_cpp.dll
echo.
echo DLL Info:
dumpbin /headers help_cpp.dll | findstr "machine" 2>nul || echo dumpbin not available, but DLL should be valid
pause