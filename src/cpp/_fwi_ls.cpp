#include <forward_solver.h>
#include <inversion_solver.h>
#include <complex_linear_system.h>
#include <mesh_generator.h>

#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(_fwi_ls, m) {
    m.def(
        "build_local_Ke",
        &fwi_ls::build_local_Ke,
        py::arg("element_id"),
        py::arg("element_points"),
        py::arg("omega"),
        py::arg("mu_field"),
        py::arg("eta_field")
    );
    m.def("build_local_f", &fwi_ls::build_local_f);

    m.def("build_extended_A", &fwi_ls::build_extended_A);

    m.def("build_connectivity_list", &fwi_ls::build_connectivity_list);

    m.def(
        "build_local_C_L",
        &fwi_ls::build_local_C_L,
        py::arg("element_id"),
        py::arg("element_points")
    );
    m.def(
        "build_local_K_L",
        &fwi_ls::build_local_K_L,
        py::arg("element_id"),
        py::arg("element_points"),
        py::arg("tau")
    );
}
