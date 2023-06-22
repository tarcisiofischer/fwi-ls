#ifndef _INVERSION_SOLVER_H
#define _INVERSION_SOLVER_H

#include <Eigen/Eigen>

namespace fwi_ls {

Eigen::Array<std::complex<double>, 4, 4> build_local_C_L(
    int element_id,
    Eigen::Ref<Eigen::Array<double, 4, 2> const> const& element_points
);

Eigen::Array<std::complex<double>, 4, 4> build_local_K_L(
    int element_id,
    Eigen::Ref<Eigen::Array<double, 4, 2> const> const& element_points,
    double tau
);

}

#endif