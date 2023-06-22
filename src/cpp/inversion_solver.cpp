#include <inversion_solver.h>

Eigen::Array<std::complex<double>, 4, 4> fwi_ls::build_local_C_L(
    int element_id,
    Eigen::Ref<Eigen::Array<double, 4, 2> const> const& element_points
)
{
    auto integration_point = Eigen::Array<double, 4, 2>();
    integration_point <<
        -1. / sqrt(3.), -1. / sqrt(3.),
        +1. / sqrt(3.), -1. / sqrt(3.),
        +1. / sqrt(3.), +1. / sqrt(3.),
        -1. / sqrt(3.), +1. / sqrt(3.);

    auto w = Eigen::Array<double, 1, 4>();
    w << 1.0, 1.0, 1.0, 1.0;

    auto N = Eigen::Array<double, 1, 4>(Eigen::Array<double, 1, 4>::Zero());
    auto C_L = Eigen::Array<std::complex<double>, 4, 4>(Eigen::Array<std::complex<double>, 4, 4>::Zero());
    for (int i = 0; i < 4; ++i) {
        auto r = integration_point(i, 0);
        auto s = integration_point(i, 1);

        N(0, 0) = (1. - r) * (1. - s) / 4.;
        N(0, 1) = (1. + r) * (1. - s) / 4.;
        N(0, 2) = (1. + r) * (1. + s) / 4.;
        N(0, 3) = (1. - r) * (1. + s) / 4.;

        auto grad_N = Eigen::Array<double, 4, 2>();
        grad_N <<
            -(1. - s) / 4., -(1. - r) / 4.,
            +(1. - s) / 4., -(1. + r) / 4.,
            +(1. + s) / 4., +(1. + r) / 4.,
            -(1. + s) / 4., +(1. - r) / 4.;

        auto J = (grad_N.transpose().matrix() * element_points.matrix()).eval();
        auto dJ = J.determinant();

        auto NT_N = (N.transpose().matrix() * N.matrix()).array();

        C_L += w(i) * NT_N * dJ;
    }

    return C_L;
}

Eigen::Array<std::complex<double>, 4, 4> fwi_ls::build_local_K_L(
    int element_id,
    Eigen::Ref<Eigen::Array<double, 4, 2> const> const& element_points,
    double tau
)
{
    auto integration_point = Eigen::Array<double, 4, 2>();
    integration_point <<
        -1. / sqrt(3.), -1. / sqrt(3.),
        +1. / sqrt(3.), -1. / sqrt(3.),
        +1. / sqrt(3.), +1. / sqrt(3.),
        -1. / sqrt(3.), +1. / sqrt(3.);

    auto w = Eigen::Array<double, 1, 4>();
    w << 1.0, 1.0, 1.0, 1.0;

    auto N = Eigen::Array<double, 1, 4>(Eigen::Array<double, 1, 4>::Zero());
    auto K_L = Eigen::Array<std::complex<double>, 4, 4>(Eigen::Array<std::complex<double>, 4, 4>::Zero());
    for (int i = 0; i < 4; ++i) {
        auto r = integration_point(i, 0);
        auto s = integration_point(i, 1);

        N(0, 0) = (1. - r) * (1. - s) / 4.;
        N(0, 1) = (1. + r) * (1. - s) / 4.;
        N(0, 2) = (1. + r) * (1. + s) / 4.;
        N(0, 3) = (1. - r) * (1. + s) / 4.;

        auto grad_N = Eigen::Array<double, 4, 2>();
        grad_N <<
            -(1. - s) / 4., -(1. - r) / 4.,
            +(1. - s) / 4., -(1. + r) / 4.,
            +(1. + s) / 4., +(1. + r) / 4.,
            -(1. + s) / 4., +(1. - r) / 4.;

        auto J = (grad_N.transpose().matrix() * element_points.matrix()).eval();
        auto dJ = J.determinant();

        auto inv_J = Eigen::Array<double, 2, 2>();
        inv_J <<
            J(1, 1), -J(0, 1),
            -J(1, 0), J(0, 0);

        auto B = (1. / dJ) * (inv_J.matrix() * grad_N.transpose().matrix()).array();
        auto BT_B = (B.transpose().matrix() * B.matrix()).array();

        K_L += w(i) * (-tau * BT_B) * dJ;
    }

    return K_L;
}
