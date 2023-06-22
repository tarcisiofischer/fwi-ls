from frequency_domain.mesh_generator import build_2d_quad_mesh
from frequency_domain.forward_case import ForwardCase
from frequency_domain.function_builders import HeavySideBasedFunctionBuilder,\
    RegionFunctionBuilder
from frequency_domain.ricker_pulse import hz_to_rads
from frequency_domain.source import ConstantSourceExpression, Source, RickerPulseSourceExpression
import numpy as np

# Sponge boundaries (Will use damping to avoid reaching the boundaries)
SPONGE_SIZE_X = 1.0  # [Km]
SPONGE_SIZE_Y = 1.0  # [Km]
DAMPING_VALUE = 1e+1

# Grid sizes
DOMAIN_SIZE_X = 1.0  # [Km]
DOMAIN_SIZE_Y = 1.0  # [Km]

# Wave velocities in each domain
C1 = 1.1
C2 = 2.2
F1 = (1.0 / np.power(C1, 2))
F2 = (1.0 / np.power(C2, 2))
# Circular inclusion parameters (C2 region)
X_C = 0.6
Y_C = 0.4
R_C = 0.2

# Source parameters
SOURCE_POSITION = (SPONGE_SIZE_X + 0.1, SPONGE_SIZE_Y + 0.2)
# Ricker pulse parameters
CENTER_FREQUENCY = 2.0  # [Hz]


def sources():
    expression = RickerPulseSourceExpression(hz_to_rads(CENTER_FREQUENCY))
    return [
        Source(f"Source", SOURCE_POSITION, expression),
    ]


def receivers():
    return []


def mesh(n=80):
    return build_2d_quad_mesh(
        size_x=DOMAIN_SIZE_X + 2 * SPONGE_SIZE_X,
        size_y=DOMAIN_SIZE_Y + 2 * SPONGE_SIZE_Y,
        nx=n,
        ny=n
    )


def mu_function():
    return RegionFunctionBuilder(overall_value=F1) \
        .set_circle_interval_value(
        center=(SPONGE_SIZE_X + X_C, SPONGE_SIZE_Y + Y_C),
        radius=R_C,
        value=F2
    ) \
        .build()


def eta_function():
    return RegionFunctionBuilder(overall_value=DAMPING_VALUE)\
        .set_rectangle_interval_value(
            start_point=(SPONGE_SIZE_X, SPONGE_SIZE_Y),
            end_point=(SPONGE_SIZE_X + DOMAIN_SIZE_X, SPONGE_SIZE_Y + DOMAIN_SIZE_Y),
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
