from scipy import interpolate
import numpy as np


def generate_discrete_field(mesh, interpolation_function):
    """
    Generate a discrete field of values, given an interpolation function.
    The generated field will be evaluated at the mesh centers.

    :param Mesh mesh:
    :param function interpolation_function:
        It is assumed to be a function generated with one of the the
        function builders.
    """
    mean_points = np.mean(mesh.points_in_elements, axis=1)

    # Assume grid is quadrilateral and orthogonal, so that all points will
    # form a 2d meshgrid, and we can take only the different values
    # for each axis in order to speedup the interpolation function
    xs = mean_points[0:mesh.ny, 0]
    ys = mean_points[::mesh.nx, 1]
    return interpolation_function(xs, ys)


class RegionFunctionBuilder():
    """
    This function builder generates a **static** property field with many
    regions. Useful for forward problems, but cannot be
    used for inversion problems, since it is not possible to provide Level Set
    information for this one.
    """
    class _RectangleRegion():
        def __init__(self, start_point, end_point, value):
            self.start_point = start_point
            self.end_point = end_point
            self.value = value
        
        def is_point_inside(self, point):
            xini, yini = self.start_point
            xend, yend = self.end_point
            x, y = point
            return xini < x < xend and yini < y < yend

    class _CircleRegion():
        def __init__(self, center, radius, value):
            self.center = center
            self.radius = radius
            self.value = value

        def is_point_inside(self, point):
            xc, yc = self.center
            r = self.radius
            x, y = point
            return (x - xc)**2 + (y - yc)**2 <= r**2

    def __init__(self, overall_value):
        self._overall_value = overall_value
        self._region_list = []

    def set_rectangle_interval_value(self, start_point, end_point, value):
        self._region_list.append(
            RegionFunctionBuilder._RectangleRegion(
                start_point,
                end_point,
                value
            )
        )
        return self

    def set_circle_interval_value(self, center, radius, value):
        self._region_list.append(
            RegionFunctionBuilder._CircleRegion(
                center,
                radius,
                value
            )
        )
        return self

    def build(self):
        def f(xs, ys):
            # TODO: Perhaps change this to use numpy vectorization. But since it
            # is used only a couple of times, maybe it is not worth it?
            result = np.empty(shape=(len(xs), len(ys)))
            result.fill(self._overall_value)
            for region in self._region_list:
                for c, x in enumerate(xs):
                    for l, y in enumerate(ys):
                        if region.is_point_inside((x, y)):
                            result[l, c] = region.value
            return result
        return f


class HeavySideBasedFunctionBuilder:
    """
    Uses the exact HeavySide function to generate a field property F0/F1 on a
    space domain of size (size_x, size_y).

    H(phi(X)) = { 0 <-> phi(X) < 0
                { 1 <-> phi(X) >= 0

    Thus, the property field is calculated using

    f(X) = H(phi(X)) * F0 + [1 - H(phi(X))] * F1

    Domain is assumed tu be a rectangle with size (size_x, size_y), starting
    at (0, 0).
    """

    def __init__(self, phi, size_x, size_y, F0, F1):
        self._phi = phi
        self._size_x = size_x
        self._size_y = size_y
        self._F0 = F0
        self._F1 = F1

    def build(self):
        H = self._compute_heavyside()

        F = H * self._F1 + (1. - H) * self._F0

        n_elems_x = self._phi.shape[0]
        n_elems_y = self._phi.shape[1]

        # Makes a linear interpolation with the desired quality. This
        # interpolation has nothing to do with the Heavy Side function, it is
        # just a way of providing a general f(X) function, so that we may ask
        # for the property at any point. This **has nothing to do** with the 
        # Geometry Mapping. ..see: SmoothHeavySideBasedFunctionBuilder
        interpolation_function = interpolate.interp2d(
            np.linspace(0.0, self._size_x, n_elems_x),
            np.linspace(0.0, self._size_y, n_elems_y),
            F
        )
        return interpolation_function

    def _compute_heavyside(self):
        H = np.empty_like(self._phi)
        H[self._phi < 0] = 0
        H[self._phi >= 0] = 1
        return H


class SmoothHeavySideBasedFunctionBuilder(HeavySideBasedFunctionBuilder):
    """
    Uses a smooth HeavySide function to generate a field property phi on
    a space domain of size (size_x, size_y), by adding a smooth function between
    phi0 and phi1. This function implements the method suggested by vandijk2013
    on "Level-set methods for structural topology optimization: a review"
    by N. P. van Dijk, K. Maute, M. Langelaar, F. van Keulen. (2013) on
    3.3 Density-based mapping

    H_hat(phi(X)) = { 0                                    <-> phi < -h
                    { -1/4(phi/h)**3 + 3/4(phi/h) + 1/2    <-> -h <= phi <= +h
                    { 1                                    <-> phi > +h
    """

    def __init__(self, phi, size_x, size_y, phi0, phi1, h):
        HeavySideBasedFunctionBuilder.__init__(self, phi, size_x, size_y, phi0, phi1)
        self._h = h

    def _compute_heavyside(self):
        h = self._h
        phi = self._phi

        H_hat = np.empty_like(phi)
        H_hat[phi < -h] = 0

        transition_ids = np.logical_and(phi >= -h, phi <= +h)
        phi_on_transition = phi[transition_ids]
        H_hat[transition_ids] = (
            (-1. / 4.) * (phi_on_transition / h)**3 + (3. / 4.) * (phi_on_transition / h) + 1/2
        )
        H_hat[phi > +h] = 1
        return H_hat
