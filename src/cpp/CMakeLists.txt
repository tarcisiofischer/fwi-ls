cmake_minimum_required(VERSION 3.10)
project(fwi_ls)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-O2")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${pybind11_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(
    _fwi_ls
    MODULE
        _fwi_ls.cpp
        complex_linear_system.h
        complex_linear_system.cpp
        forward_solver.h
        forward_solver.cpp
        inversion_solver.h
        inversion_solver.cpp
        mesh_generator.h
        mesh_generator.cpp
)

target_link_libraries(
    _fwi_ls
    PRIVATE
        pybind11::module
        Eigen3::Eigen
)
set_target_properties(
    _fwi_ls
    PROPERTIES
        PREFIX ""
# TODO: Automatic setup
        SUFFIX ".so" # Linux
#        SUFFIX ".pyd" # Windows
)
