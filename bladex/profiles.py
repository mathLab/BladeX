"""
Derived module from profilebase.py to provide the airfoil coordinates.
"""

import numpy as np
from .profilebase import ProfileBase


class CustomProfile(ProfileBase):
    """
    Provide custom profile for the airfoil coordinates.

    :param numpy.ndarray xup: 1D array that contains the X-components of the airfoil's upper surface
    :param numpy.ndarray xdown: 1D array that contains the X-components of the airfoil's lower surface
    :param numpy.ndarray yup: 1D array that contains the Y-components of the airfoil's upper surface
    :param numpy.ndarray ydown: 1D array that contains the Y-components of the airfoil's lower surface
    """

    def __init__(self, xdown, xup, ydown, yup):
        self.xdown_coordinates = xdown
        self.xup_coordinates = xup
        self.ydown_coordinates = ydown
        self.yup_coordinates = yup
        self._check_coordinates()

    def _check_coordinates(self):
        """
        Private method that checks whether the airfoil coordinates defined are provided correctly.

        We note that each array of coordinates must be consistent with the other arrays.
        The upper and lower surfaces should start from exactly the same point, the leading edge,
        and proceed on the way till the trailing edge. The trailing edge might have a non-zero thickness as in the case of some NACA-airfoils. In case of an open trailing edge, 
        the average coordinate between upper and lower part is taken as the unique value.

        :raises ValueError: if either xup, xdown, yup, ydown is None
        :raises ValueError: if the 1D arrays xup, yup or xdown, ydown do not have the same length
        :raises ValueError: if array yup not greater than or equal array ydown element-wise
        :raises ValueError: if xdown[0] != xup[0] or ydown[0] != yup[0] or xdown[-1] != xup[-1]
        """
        if self.xup_coordinates is None:
            raise ValueError(
                'object "xup_coordinates" refers to an empty array.')
        if self.xdown_coordinates is None:
            raise ValueError(
                'object "xdown_coordinates" refers to an empty array.')
        if self.yup_coordinates is None:
            raise ValueError(
                'object "yup_coordinates" refers to an empty array.')
        if self.ydown_coordinates is None:
            raise ValueError(
                'object "ydown_coordinates" refers to an empty array.')

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

        # Therefore the arrays xup_coordinates and yup_coordinates must have the same length = N,
        # Same holds for the arrays xdown_coordinates and ydown_coordinates.
        if self.xup_coordinates.shape != self.yup_coordinates.shape:
            raise ValueError(
                'Arrays xup_coordinates and yup_coordinates do not have the same shape.'
            )
        if self.xdown_coordinates.shape != self.ydown_coordinates.shape:
            raise ValueError(
                'Arrays xdown_coordinates and ydown_coordinates do not have the same shape.'
            )

        # - The condition yup_coordinates >= ydown_coordinates must be satisfied element-wise to the whole elements in the mentioned arrays.
        if not all(
                np.greater_equal(self.yup_coordinates, self.ydown_coordinates)):
            raise ValueError(
                'yup_coordinates !>= ydown_coordinates elementwise.')

        if not self.xdown_coordinates[0] == self.xup_coordinates[0]:
            raise ValueError(
                'The condition (xdown_coordinates[0]=xup_coordinates[0]) is not satisfied.'
            )
        if not self.ydown_coordinates[0] == self.yup_coordinates[0]:
            raise ValueError(
                'The condition (ydown_coordinates[0]=yup_coordinates[0]) is not satisfied.'
            )
        if not self.xdown_coordinates[-1] == self.xup_coordinates[-1]:
            raise ValueError(
                'The condition (xdown_coordinates[-1]=xup_coordinates[-1]) is not satisfied.'
            )


class NacaProfile(ProfileBase):
    """
    Generate 4- and 5-digit NACA profiles.

    The NACA airfoils are airfoil shapes for aircraft wings developed by the National Advisory Committee for Aeronautics (NACA).
    The shape of the NACA airfoils is described using a series of digits following the word "NACA". The parameters in the numerical
    code can be entered into equations to precisely generate the cross-section of the airfoil and calculate its properties.

    The NACA four-digit series describes airfoil by the format MPTT, where:
        M/100: indicates the maximum camber in percentage, with respect to the chord length.
        P/10: indicates the location of the maximum camber measured from the leading edge. The location is normalized by the chord length.
        TT/100: the maximum thickness as fraction of the chord length.

    The profile 00TT refers to a symmetrical NACA airfoil.

    The NACA five-digit series describes more complex airfoil shapes. Its format is: LPSTT, where:
        L: the theoretical optimum lift coefficient at ideal angle-of-attack = 0.15*L
        P: the x-coordinate of the point of maximum camber (max camber at x = 0.05*P)
        S: indicates whether the camber is simple (S=0) or reflex (S=1)
        TT/100: the maximum thickness in percent of chord, as in a four-digit NACA airfoil code

    References:
    Moran, Jack (2003). An introduction to theoretical and computational aerodynamics. Dover. p. 7. ISBN 0-486-42879-6.
    Abbott, Ira (1959). Theory of Wing Sections: Including a Summary of Airfoil Data. New York: Dover Publications. p. 115. ISBN 978-0486605869.

    :param string digits: 4 or 5 digits that describes the NACA profile
    :param int n_points: number of discrete points that represents the airfoil profile. Default value is 240
    :raises ValueError: if n_points is not positive
    :raises TypeError: if n_points is not of type int
    :raises SyntaxError: if digits is not a string
    :raises Exception: if digits is not of length 4 or 5
    """

    def __init__(self, digits, n_points=240):
        self.digits = digits
        self.n_points = n_points
        self._check_args()
        self._generate_coordinates()

    def _check_args(self):
        """
        Private method to check that the number of the airfoil discrete points is a positive integer.
        """
        if not isinstance(self.n_points, int):
            raise TypeError('n_points must be of type integer.')
        if self.n_points < 0:
            raise ValueError('n_points must be positive.')

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
        Private method that generates the coordinates of the NACA 4 or 5 digits airfoil profile. 
        The method assumes a zero-thickness trailing edge, and no half-cosine spacing.
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
            a4 = -0.1036  # zero thickness TE

            x = np.linspace(0.0, 1.0, num=self.n_points + 1)

            yt = 5 * t * (a0 * np.sqrt(x) + a1 * x + a2 * np.power(x, 2) +
                          a3 * np.power(x, 3) + a4 * np.power(x, 4))

            if p == 0:
                self.xup_coordinates = np.linspace(
                    0.0, 1.0, num=self.n_points + 1)
                self.yup_coordinates = yt
                self.xdown_coordinates = np.linspace(
                    0.0, 1.0, num=self.n_points + 1)
                self.ydown_coordinates = -yt
            else:
                xc1 = np.asarray([xx for xx in x if xx <= p])
                xc2 = np.asarray([xx for xx in x if xx > p])
                yc1 = m / np.power(p, 2) * xc1 * (2 * p - xc1)
                yc2 = m / np.power(1 - p, 2) * (1 - 2 * p + xc2) * (1 - xc2)
                zc = np.append(yc1, yc2)

                dyc1_dx = m / np.power(p, 2) * (2 * p - 2 * xc1)
                dyc2_dx = m / np.power(1 - p, 2) * (2 * p - 2 * xc2)
                dyc_dx = np.append(dyc1_dx, dyc2_dx)

                theta = np.arctan(dyc_dx)

                self.xup_coordinates = np.asarray(
                    [xx - yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)])
                self.yup_coordinates = np.asarray(
                    [xx + yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)])
                self.xdown_coordinates = np.asarray(
                    [xx + yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)])
                self.ydown_coordinates = np.asarray(
                    [xx - yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)])

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

            yt = 5 * t * (a0 * np.sqrt(x) + a1 * x + a2 * np.power(x, 2) +
                          a3 * np.power(x, 3) + a4 * np.power(x, 4))

            P = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
            M = np.array([0.0580, 0.1260, 0.2025, 0.2900, 0.3910])
            K = np.array([361.4, 51.64, 15.957, 6.643, 3.230])

            if p == 0:
                self.xup_coordinates = np.linspace(
                    0.0, 1.0, num=self.n_points + 1)
                self.yup_coordinates = yt
                self.xdown_coordinates = np.linspace(
                    0.0, 1.0, num=self.n_points + 1)
                self.ydown_coordinates = -yt
            else:
                m = self._cubic_spline_interpolation(P, M, [p])
                k1 = self._cubic_spline_interpolation(M, K, [m])
                xc1 = np.asarray([xx for xx in x if xx <= p])
                xc2 = np.asarray([xx for xx in x if xx > p])
                yc1 = k1 / 6.0 * (np.power(xc1, 3) - 3 * m * np.power(xc1, 2) +
                                  np.power(m, 2) * (3 - m) * xc1)
                yc2 = k1 / 6.0 * np.power(m, 3) * (1 - xc2)
                yc = np.append(yc1, yc2)
                zc = cld / 0.3 * yc

                dyc1_dx = cld / 0.3 * (1.0 / 6.0) * k1 * (
                    3 * np.power(xc1, 2) - 6 * m * xc1 + np.power(m, 2) *
                    (3 - m))
                dyc2_dx = np.tile(cld / 0.3 * (1.0 / 6.0) * k1 * np.power(m, 3),
                                  len(xc2))

                dyc_dx = np.append(dyc1_dx, dyc2_dx)
                theta = np.arctan(dyc_dx)

                self.xup_coordinates = np.asarray(
                    [xx - yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)])
                self.yup_coordinates = np.asarray(
                    [xx + yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)])
                self.xdown_coordinates = np.asarray(
                    [xx + yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)])
                self.ydown_coordinates = np.asarray(
                    [xx - yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)])

        else:
            raise Exception
