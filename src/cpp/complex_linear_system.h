#ifndef _COMPLEX_LINEAR_SYSTEM_H
#define _COMPLEX_LINEAR_SYSTEM_H

#include <tuple>
#include <Eigen/Eigen>

namespace fwi_ls {
    std::tuple<
        Eigen::Array<int, Eigen::Dynamic, 1>,
        Eigen::Array<int, Eigen::Dynamic, 1>,
        Eigen::Array<double, Eigen::Dynamic, 1>
    > build_extended_A(
        Eigen::Ref<Eigen::Array<int, Eigen::Dynamic, 1> const> const& A_row,
        Eigen::Ref<Eigen::Array<int, Eigen::Dynamic, 1> const> const& A_col,
        Eigen::Ref<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> const> const& A_data,
        int n_rows,
        int n_cols
    );
}

#endif
