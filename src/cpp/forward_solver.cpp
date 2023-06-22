#include <forward_solver.h>

Eigen::Array<std::complex<double>, 4, 4> fwi_ls::build_local_Ke(
    int element_id,
    Eigen::Ref<Eigen::Array<double, 4, 2> const> const& element_points,
    double omega,
    Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1> const> const& mu_field,
    Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1> const> const& eta_field
)
{
    auto const& mu = mu_field(element_id);
    auto const& eta = eta_field(element_id);

    auto integration_point = Eigen::Array<double, 4, 2>();
    integration_point <<
        -1. / sqrt(3.), -1. / sqrt(3.),
        +1. / sqrt(3.), -1. / sqrt(3.),
        +1. / sqrt(3.), +1. / sqrt(3.),
        -1. / sqrt(3.), +1. / sqrt(3.);

    auto w = Eigen::Array<double, 1, 4>();
    w << 1.0, 1.0, 1.0, 1.0;

    auto N = Eigen::Array<double, 1, 4>(Eigen::Array<double, 1, 4>::Zero());
    auto M_hat = Eigen::Array<double, 4, 4>(Eigen::Array<double, 4, 4>::Zero());
    auto C_hat = Eigen::Array<std::complex<double>, 4, 4>(Eigen::Array<std::complex<double>, 4, 4>::Zero());
    auto K_hat = Eigen::Array<double, 4, 4>(Eigen::Array<double, 4, 4>::Zero());
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

        // TODO: I don't know if the following operations are fast. Must check
        // later.
        auto J = (grad_N.transpose().matrix() * element_points.matrix()).eval();
        auto dJ = J.determinant();

        auto inv_J = Eigen::Array<double, 2, 2>();
        inv_J <<
            J(1, 1), -J(0, 1),
            -J(1, 0), J(0, 0);

        auto B = (1. / dJ) * (inv_J.matrix() * grad_N.transpose().matrix()).array();
        auto NT_N = (N.transpose().matrix() * N.matrix()).array();
        auto BT_B = (B.transpose().matrix() * B.matrix()).array();

        M_hat += w(i) * -(omega * omega) * mu * NT_N * dJ;
        C_hat += w(i) * std::complex<double>(0.0, 1.0) * omega * eta * NT_N * dJ;
        K_hat += w(i) * BT_B * dJ;
    }
    auto K = Eigen::Array<std::complex<double>, 4, 4>(M_hat + C_hat + K_hat);

    return K;
}

Eigen::Array<double, 4, 1> fwi_ls::build_local_f(
    Eigen::Array<double, 4, 2> element_points,
    Eigen::Array<double, 4, 1> S_e
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
    auto f = Eigen::Array<double, 4, 4>(Eigen::Array<double, 4, 4>::Zero());
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

        // TODO: I don't know if the following operations are fast. Must check
        auto J = (grad_N.transpose().matrix() * element_points.matrix()).eval();
        auto dJ = J.determinant();

        auto NT_N = (N.transpose().matrix() * N.matrix()).array();
        f += w(i) * NT_N * dJ;
    }
    return -(f.matrix() * S_e.matrix()).array();
}
