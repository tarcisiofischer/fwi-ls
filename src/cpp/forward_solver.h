#ifndef _FORWARD_SOLVER_H
#define _FORWARD_SOLVER_H

#include <Eigen/Eigen>

namespace fwi_ls {

Eigen::Array<std::complex<double>, 4, 4> build_local_Ke(
    int element_id,
    Eigen::Ref<Eigen::Array<double, 4, 2> const> const& element_points,
    double omega,
    Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1> const> const& mu_field,
    Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1> const> const& eta_field
);

Eigen::Array<double, 4, 1> build_local_f(
    Eigen::Array<double, 4, 2> element_points,
    Eigen::Array<double, 4, 1> S_e
);

}

#endif