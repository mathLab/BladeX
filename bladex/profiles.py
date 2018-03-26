"""
"""
from math import sqrt
import numpy as np
from .profilebase import ProfileBase


class CustomProfile(ProfileBase):
    """
    TO DOC
    """

    def __init__(self, xdown, xup, ydown, yup):
        self.xdown_coordinates = xdown
        self.xup_coordinates = xup
        self.ydown_coordinates = ydown
        self.yup_coordinates = yup
        self._check_coordinates()

    def _check_coordinates(self):
        """
        TO DOC
        TODO: 
        - add a check on single coordinates of ydown and yup. yup has to be ABOVE ydown componentwise.
        - check on xdown[0] and xup[0]: they have to be equal. Same for [-1].
        """
        assert self.xup_coordinates is not None, "object 'xup_coordinates' refers to an empty array."
        assert self.xdown_coordinates is not None, "object 'xdown_coordinates' refers to an empty array."
        assert self.yup_coordinates is not None, "object 'yup_coordinates' refers to an empty array."
        assert self.ydown_coordinates is not None, "object 'ydown_coordinates' refers to an empty array."

        if not isinstance(self.xup_coordinates, np.ndarray):
            self.xup_coordinates = np.asarray(self.xup_coordinates, dtype=float)
        if not isinstance(self.xdown_coordinates, np.ndarray):
            self.xdown_coordinates = np.asarray(
                self.xdown_coordinates, dtype=float)
        if not isinstance(self.yup_coordinates, np.ndarray):
            self.yup_coordinates = np.asarray(self.yup_coordinates, dtype=float)
        if not isinstance(self.ydown_coordinates, np.ndarray):
            self.ydown_coordinates = np.asarray(
                self.ydown_coordinates, dtype=float)
        if self.xup_coordinates.shape != self.yup_coordinates.shape or self.xdown_coordinates.shape != self.ydown_coordinates.shape:
            raise ValueError(
                'Arrays {xup_coordinates, yup_coordinates} or {xdown_coordinates, ydown_coordinates} do not have the same shape.'
            )

        if not all(
                np.greater_equal(self.yup_coordinates, self.ydown_coordinates)):
            raise ValueError(
                'yup_coordinates !>= ydown_coordinates elementwise.')

        # FIXED
        # TODO: split the checks, same above
        if (not self.xdown_coordinates[0] == self.xup_coordinates[0]) or (
                not self.xdown_coordinates[-1] == self.xup_coordinates[-1]) or (
                    not self.ydown_coordinates[0] == self.yup_coordinates[0]):
            raise ValueError(
                'One of the following conditions is not satisfied: (xdown_coordinates[0]=xup_coordinates[0]) or (xdown_coordinates[-1]=xup_coordinates[-1]) or (ydown_coordinates[0]=yup_coordinates[0])'
            )


class NacaProfile(ProfileBase):
    """
    TO DOC
    the 5 digits does not work properly. We can think to remove it in the near future.
    """

    def __init__(self, digits, n_points=240):
        self.digits = digits
        self.n_points = n_points
        self._generate_coordinates()

    @staticmethod
    def _cubic_spline_interpolation(xa, ya, query_points):
        """
        A cubic spline interpolation on a given set of points (x, y)
        Recalculates everything on every call which is far from efficient but does the job for now
        should eventually be replaced by an external helper class
        """
        # Adapted from:
        # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
        # ISBN 0-521-43108-5, page 113, section 3.3.
        n_points = len(xa)
        u, y2 = [0] * n_points, [0] * n_points

        for i in range(1, n_points - 1):
            # This is the decomposition loop of the tridiagonal algorithm.
            # y2 and u are used for temporary storage of the decomposed factors.
            wx = xa[i + 1] - xa[i - 1]
            sig = (xa[i] - xa[i - 1]) / wx
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            ddydx = (ya[i + 1] - ya[i]) / (xa[i + 1] - xa[i]) - (
                ya[i] - ya[i - 1]) / (xa[i] - xa[i - 1])
            u[i] = (6.0 * ddydx / wx - sig * u[i - 1]) / p

        # This is the backsubstitution loop of the tridiagonal algorithm
        for i in range(n_points - 2, -1, -1):
            y2[i] = y2[i] * y2[i + 1] + u[i]

        # bisection. This is optimal if sequential calls to this
        # routine are at random values of x. If sequential calls
        # are in order, and closely spaced, one would do better
        # to store previous values of klo and khi
        klo = 0
        khi = n_points - 1

        while (khi - klo > 1):
            k = (khi + klo) >> 1
            if (xa[k] > query_points[0]):
                khi = k
            else:
                klo = k

        h = xa[khi] - xa[klo]
        a = (xa[khi] - query_points[0]) / h
        b = (query_points[0] - xa[klo]) / h

        # Cubic spline polynomial is now evaluated
        result = a * ya[klo] + b * ya[khi] + (
            (a * a * a - a) * y2[klo] +
            (b * b * b - b) * y2[khi]) * (h * h) / 6.0
        return result

    def _generate_coordinates(self):
        """
        TO DOC
        Works always as finite_TE=False, half_cosine_spacing=False
        """
        if len(self.digits) == 4:
            # Returns n+1 points in [0 1] for the given 4 digit NACA number string
            m = float(self.digits[0]) / 100.0
            p = float(self.digits[1]) / 10.0
            t = float(self.digits[2:]) / 100.0

            a0 = +0.2969
            a1 = -0.1260
            a2 = -0.3516
            a3 = +0.2843
            a4 = -0.1036  # zero thick TE

            x = np.linspace(0.0, 1.0, num=self.n_points + 1)

            yt = [
                5 * t * (a0 * sqrt(xx) + a1 * xx + a2 * pow(xx, 2) +
                         a3 * pow(xx, 3) + a4 * pow(xx, 4)) for xx in x
            ]

            if p == 0:
                xu = np.linspace(0.0, 1.0, num=self.n_points + 1)
                yu = yt
                xl = np.linspace(0.0, 1.0, num=self.n_points + 1)
                yl = [-xx for xx in yt]
            else:
                xc1 = [xx for xx in x if xx <= p]
                xc2 = [xx for xx in x if xx > p]
                yc1 = [m / pow(p, 2) * xx * (2 * p - xx) for xx in xc1]
                yc2 = [
                    m / pow(1 - p, 2) * (1 - 2 * p + xx) * (1 - xx)
                    for xx in xc2
                ]
                zc = yc1 + yc2

                dyc1_dx = [m / pow(p, 2) * (2 * p - 2 * xx) for xx in xc1]
                dyc2_dx = [m / pow(1 - p, 2) * (2 * p - 2 * xx) for xx in xc2]
                dyc_dx = dyc1_dx + dyc2_dx

                theta = [atan(xx) for xx in dyc_dx]

                xu = [xx - yy * sin(zz) for xx, yy, zz in zip(x, yt, theta)]
                yu = [xx + yy * cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

                xl = [xx + yy * sin(zz) for xx, yy, zz in zip(x, yt, theta)]
                yl = [xx - yy * cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

            self.xup_coordinates = np.asarray(xu)
            self.xdown_coordinates = np.asarray(xl)
            self.yup_coordinates = np.asarray(yu)
            self.ydown_coordinates = np.asarray(yl)

        elif len(self.digits) == 5:
            # Returns n+1 points in [0 1] for the given 5 digit NACA number string
            naca1 = float(self.digits[0])
            naca23 = float(self.digits[1:3])
            naca45 = float(self.digits[3:])

            cld = naca1 * (3.0 / 2.0) / 10.0
            p = 0.5 * naca23 / 100.0
            t = naca45 / 100.0

            a0 = +0.2969
            a1 = -0.1260
            a2 = -0.3516
            a3 = +0.2843
            a4 = -0.1036  # For zero thickness trailing edge

            x = np.linspace(0.0, 1.0, num=self.n_points + 1)

            yt = [
                5 * t * (a0 * sqrt(xx) + a1 * xx + a2 * pow(xx, 2) +
                         a3 * pow(xx, 3) + a4 * pow(xx, 4)) for xx in x
            ]

            P = [0.05, 0.1, 0.15, 0.2, 0.25]
            M = [0.0580, 0.1260, 0.2025, 0.2900, 0.3910]
            K = [361.4, 51.64, 15.957, 6.643, 3.230]

            if p == 0:
                xu = np.linspace(0.0, 1.0, num=self.n_points + 1)
                yu = yt
                xl = np.linspace(0.0, 1.0, num=self.n_points + 1)
                yl = [-yy for yy in yt]
            else:
                m = self._cubic_spline_interpolation(P, M, [p])
                k1 = self._cubic_spline_interpolation(M, K, [m])
                xc1 = [xx for xx in x if xx <= p]
                xc2 = [xx for xx in x if xx > p]
                yc1 = [
                    k1 / 6.0 * (pow(xx, 3) - 3 * m * pow(xx, 2) + pow(m, 2) *
                                (3 - m) * xx) for xx in xc1
                ]
                yc2 = [k1 / 6.0 * pow(m, 3) * (1 - xx) for xx in xc2]
                zc = [cld / 0.3 * xx for xx in yc1 + yc2]

                dyc1_dx = [
                    cld / 0.3 * (1.0 / 6.0) * k1 *
                    (3 * pow(xx, 2) - 6 * m * xx + pow(m, 2) * (3 - m))
                    for xx in xc1
                ]
                dyc2_dx = [cld / 0.3 * (1.0 / 6.0) * k1 * pow(m, 3)] * len(xc2)

                dyc_dx = dyc1_dx + dyc2_dx
                theta = [atan(xx) for xx in dyc_dx]

                xu = [xx - yy * sin(zz) for xx, yy, zz in zip(x, yt, theta)]
                yu = [xx + yy * cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

                xl = [xx + yy * sin(zz) for xx, yy, zz in zip(x, yt, theta)]
                yl = [xx - yy * cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

            self.xup_coordinates = np.asarray(xu)
            self.xdown_coordinates = np.asarray(xl)
            self.yup_coordinates = np.asarray(yu)
            self.ydown_coordinates = np.asarray(yl)

        else:
            raise Exception
