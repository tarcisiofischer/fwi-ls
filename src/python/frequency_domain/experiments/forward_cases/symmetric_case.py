from frequency_domain.mesh_generator import build_2d_quad_mesh
from frequency_domain.forward_case import ForwardCase
from frequency_domain.function_builders import HeavySideBasedFunctionBuilder,\
    RegionFunctionBuilder
from frequency_domain.receiver import build_receivers
from frequency_domain.source import ConstantSourceExpression, Source
import numpy as np

# Grid sizes
DOMAIN_SIZE_X = 5.0  # [Km]
DOMAIN_SIZE_Y = 5.0  # [Km]

# Sponge boundaries (Will use damping to avoid reaching the boundaries)
DAMPING_VALUE = 2e+1

C1 = 1.0
C2 = 3.5
F1 = (1.0 / np.power(C1, 2))
F2 = (1.0 / np.power(C2, 2))

# Parameters for inversion
OMEGA_I = 2.0
OMEGA_F = 6.0
N_OMEGAS = 4


def sources():
    expression = ConstantSourceExpression(2e-2)
    return [
        Source(f"Source 1", (1.0, 2.5), expression),
        Source(f"Source 2", (4.0, 2.5), expression),
        Source(f"Source 3", (2.5, 1.0), expression),
        Source(f"Source 4", (2.5, 4.0), expression),
    ]


def receivers():
    return build_receivers(
        start_position=(1.1, 1.75),
        end_position=(1.1, 3.25),
        amount=6,
        prefix="left_",
    ) + build_receivers(
        start_position=(3.9, 1.75),
        end_position=(3.9, 3.25),
        amount=6,
        prefix="right_",
    ) + build_receivers(
        start_position=(1.75, 1.1),
        end_position=(3.25, 1.1),
        amount=6,
        prefix="top",
    ) + build_receivers(
        start_position=(1.75, 3.9),
        end_position=(3.25, 3.9),
        amount=6,
        prefix="bottom",
    )


def mesh(n=60):
    return build_2d_quad_mesh(
        size_x=DOMAIN_SIZE_X,
        size_y=DOMAIN_SIZE_Y,
        nx=n,
        ny=n
    )


def mu_function():
    # Level set phi function (discrete)
    phi = -1.0 * np.ones(shape=(60 + 1, 60 + 1))
    phi[25:36, 25:36] = 1.0

    return HeavySideBasedFunctionBuilder(
        phi,
        DOMAIN_SIZE_X,
        DOMAIN_SIZE_Y,
        F1,
        F2
    ).build()


def eta_function():
    return RegionFunctionBuilder(overall_value=DAMPING_VALUE)\
        .set_rectangle_interval_value(
            start_point=(1.0, 1.0),
            end_point=(4.0, 4.0),
            value=0.0
        )\
        .build()


def build_forward_case(source, omega, n=60):
    return ForwardCase(
        mesh=mesh(n),
        omega=omega,
        mu_function=mu_function(),
        eta_function=eta_function(),
        source=source
    )
