from frequency_domain.mesh_generator import build_2d_quad_mesh
from frequency_domain.forward_case import ForwardCase
from frequency_domain.function_builders import RegionFunctionBuilder
from frequency_domain.receiver import build_receivers
from frequency_domain.ricker_pulse import hz_to_rads
from frequency_domain.source import RickerPulseSourceExpression, Source
import numpy as np

# Grid sizes
DOMAIN_SIZE_X = 1.0  # [Km]
DOMAIN_SIZE_Y = 0.5  # [Km]

# Sponge boundaries (Will use damping to avoid reaching the boundaries)
SPONGE_SIZE_X = 0.5  # [Km]
SPONGE_SIZE_Y = 0.5  # [Km]
DAMPING_VALUE = 1e+1

# Wave velocities in each domain
C1 = 1.5  # [Km/s]
C2 = 2.5  # [Km/s]
F1 = (1.0 / np.power(C1, 2))
F2 = (1.0 / np.power(C2, 2))
# "Plus" inclusion parameters
pa_i = (SPONGE_SIZE_X + 0.6, SPONGE_SIZE_Y + 0.3)
pa_f = (SPONGE_SIZE_X + 0.9, SPONGE_SIZE_Y + 0.35)
pb_i = (SPONGE_SIZE_X + 0.725, SPONGE_SIZE_Y + 0.2)
pb_f = (SPONGE_SIZE_X + 0.775, SPONGE_SIZE_Y + 0.45)

# Sources parameters
SOURCE_X_I = 0.042
SOURCE_DX = 0.067
SOURCE_Y = 0.092
N_SOURCES = 10

# Ricker pulse parameters
CENTER_FREQUENCY = 2.0  # [rad/s]

# Parameters for inversion
# Parameters for Ricker pulse integration
OMEGA_I = 2.0
OMEGA_F = 30.0
N_OMEGAS = 16
# Number of receivers. For this case, only one side of the domain will contain them
N_RECEIVERS = 30


def sources():
    expression = RickerPulseSourceExpression(hz_to_rads(CENTER_FREQUENCY))
    return [
        Source(
            f"Source {i}",
            (SPONGE_SIZE_X + SOURCE_X_I + i * SOURCE_DX, SPONGE_SIZE_Y + SOURCE_Y),
            expression
        ) for i in range(N_SOURCES)
    ]


def receivers():
    # Move the receivers a bit to inside the domain, to avoid touching the
    # damping area.
    y_correction = 0.02
    return build_receivers(
        start_position=(SPONGE_SIZE_X, SPONGE_SIZE_Y + y_correction),
        end_position=(SPONGE_SIZE_X + DOMAIN_SIZE_X, SPONGE_SIZE_Y + y_correction),
        amount=N_RECEIVERS,
    )


def mesh(n=60):
    return build_2d_quad_mesh(
        size_x=DOMAIN_SIZE_X + 2 * SPONGE_SIZE_X,  # [Km]
        size_y=DOMAIN_SIZE_Y + 2 * SPONGE_SIZE_Y,  # [Km]
        nx=n,
        ny=n
    )


def mu_function():
    return RegionFunctionBuilder(overall_value=F1) \
        .set_rectangle_interval_value(
        start_point=pa_i,
        end_point=pa_f,
        value=F2
    ) \
        .set_rectangle_interval_value(
        start_point=pb_i,
        end_point=pb_f,
        value=F2
    ) \
        .build()


def eta_function():
    return RegionFunctionBuilder(overall_value=DAMPING_VALUE) \
        .set_rectangle_interval_value(
        start_point=(SPONGE_SIZE_X, SPONGE_SIZE_Y),
        end_point=(SPONGE_SIZE_X + DOMAIN_SIZE_X, SPONGE_SIZE_Y + DOMAIN_SIZE_Y),
        value=0.0
    ) \
        .build()


def build_forward_case(source, omega, n=60):
    return ForwardCase(
        mesh=mesh(n),
        omega=omega,
        mu_function=mu_function(),
        eta_function=eta_function(),
        source=source
    )
