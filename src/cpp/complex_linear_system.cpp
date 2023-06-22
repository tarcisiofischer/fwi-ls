#include <complex_linear_system.h>

std::tuple<
    Eigen::Array<int, Eigen::Dynamic, 1>,
    Eigen::Array<int, Eigen::Dynamic, 1>,
    Eigen::Array<double, Eigen::Dynamic, 1>
> fwi_ls::build_extended_A(
    Eigen::Ref<Eigen::Array<int, Eigen::Dynamic, 1> const> const& A_row,
    Eigen::Ref<Eigen::Array<int, Eigen::Dynamic, 1> const> const& A_col,
    Eigen::Ref<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> const> const& A_data,
    int n_rows,
    int n_cols
)
{
    auto size = 4 * A_row.rows();

    auto big_A_coo_i = Eigen::Array<int, Eigen::Dynamic, 1>(Eigen::Array<int, Eigen::Dynamic, 1>::Zero(size, 1));
    auto big_A_coo_j = Eigen::Array<int, Eigen::Dynamic, 1>(Eigen::Array<int, Eigen::Dynamic, 1>::Zero(size, 1));
    auto big_A_coo_data = Eigen::Array<double, Eigen::Dynamic, 1>(Eigen::Array<double, Eigen::Dynamic, 1>::Zero(size, 1));

    for (int idx = 0; idx < A_row.rows(); ++idx) {
        auto i = A_row(idx, 0);
        auto j = A_col(idx, 0);
        auto data = A_data(idx, 0);
    
        big_A_coo_i(4 * idx, 0) = i;
        big_A_coo_j(4 * idx, 0) = j;
        big_A_coo_data(4 * idx, 0) = data.real();
    
        big_A_coo_i(4 * idx + 1, 0) = i + n_rows;
        big_A_coo_j(4 * idx + 1, 0) = j;
        big_A_coo_data(4 * idx + 1, 0) = data.imag();

        big_A_coo_i(4 * idx + 2 , 0) = i;
        big_A_coo_j(4 * idx + 2, 0) = j + n_cols;
        big_A_coo_data(4 * idx + 2, 0) = -data.imag();
    
        big_A_coo_i(4 * idx + 3, 0) = i + n_rows;
        big_A_coo_j(4 * idx + 3, 0) = j + n_cols;
        big_A_coo_data(4 * idx + 3, 0) = data.real();
    }

    return std::make_tuple(big_A_coo_i, big_A_coo_j, big_A_coo_data);
}
