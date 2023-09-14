"""
Derived module from profilebase.py to provide the airfoil coordinates for
standard Naca profiles.
"""
from scipy.interpolate import splev, splrep
import numpy as np
from .profileinterface import ProfileInterface

class NacaProfile(ProfileInterface):
    """
    Generate 4- and 5-digit NACA profiles.

    The NACA airfoils are airfoil shapes for aircraft wings developed by the
    National Advisory Committee for Aeronautics (NACA). The shape of the NACA
    airfoils is described using a series of digits following the word "NACA".
    The parameters in the numerical code can be entered into equations to
    precisely generate the cross-section of the airfoil and calculate its
    properties.

    The NACA four-digit series describes airfoil by the format MPTT, where:

        - M/100: indicates the maximum camber in percentage, with respect to the
          chord length.

        - P/10: indicates the location of the maximum camber measured from the
          leading edge. The location is normalized by the chord length.

        - TT/100: the maximum thickness as fraction of the chord length.

    The profile 00TT refers to a symmetrical NACA airfoil.

    The NACA five-digit series describes more complex airfoil shapes.
    Its format is: LPSTT, where:

        - L: the theoretical optimum lift coefficient at ideal
          angle-of-attack = 0.15*L

        - P: the x-coordinate of the point of maximum camber
          (max camber at x = 0.05*P)

        - S: indicates whether the camber is simple (S=0) or reflex (S=1)
          TT/100: the maximum thickness in percent of chord, as in a four-digit
          NACA airfoil code

    References:

    - Moran, Jack (2003). An introduction to theoretical and computational
      aerodynamics. Dover. p. 7. ISBN 0-486-42879-6.

    - Abbott, Ira (1959). Theory of Wing Sections: Including a Summary of
      Airfoil Data. New York: Dover Publications. p. 115. ISBN 978-0486605869.

    :param str digits: 4 or 5 digits that describes the NACA profile
    :param int n_points: number of discrete points that represents the
        airfoil profile. Default value is 240
    :param bool cosine_spacing: if True, then a cosine spacing is used for the
        airfoil coordinate distribution, otherwise linear spacing is used.
        Default value is True
    :raises ValueError: if n_points is not positive
    :raises TypeError: if n_points is not of type int
    :raises SyntaxError: if digits is not a string
    :raises Exception: if digits is not of length 4 or 5
    """

    def __init__(self, digits, n_points=240, cosine_spacing=True):
        super(NacaProfile, self).__init__()
        self.digits = digits
        self.n_points = n_points
        self.cosine_spacing = cosine_spacing
        self._check_args()
        self.generate_coordinates()
        self.generate_parameters(convention='british')

    def _check_args(self):
        """
        Private method to check that the number of the airfoil discrete points
        is a positive integer.
        """
        if not isinstance(self.digits, str):
            raise TypeError('digits must be of type string.')
        if isinstance(self.n_points, float):
            self.n_points = int(self.n_points)
        if not isinstance(self.n_points, int):
            raise TypeError('n_points must be of type integer.')
        if self.n_points < 0:
            raise ValueError('n_points must be positive.')

    def generate_parameters(self, convention='british'):
        return super().generate_parameters(convention)

    def generate_coordinates(self):
        """
        Private method that generates the coordinates of the NACA 4 or 5 digits
        airfoil profile. The method assumes a zero-thickness trailing edge, and
        no half-cosine spacing.
        """
        a0 = +0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = +0.2843
        a4 = -0.1036  # zero thickness TE

        x = np.linspace(0.0, 1.0, num=self.n_points)

        if len(self.digits) == 4:
            # Returns n+1 points in [0 1] for the given 4-digits NACA string
            m = float(self.digits[0]) / 100.0
            p = float(self.digits[1]) / 10.0
            t = float(self.digits[2:]) / 100.0

            # half-thickness distribution
            yt = 5 * t * (a0 * np.sqrt(x) + a1 * x + a2 * np.power(x, 2) +
                          a3 * np.power(x, 3) + a4 * np.power(x, 4))

            if p == 0:
                # Symmetric foil
                self.xup_coordinates = np.linspace(0.0, 1.0, num=self.n_points)
                self.yup_coordinates = yt
                self.xdown_coordinates = np.linspace(
                    0.0, 1.0, num=self.n_points)
                self.ydown_coordinates = -yt
            else:
                # Cambered foil
                xc1 = np.asarray([xx for xx in x if xx <= p])
                xc2 = np.asarray([xx for xx in x if xx > p])
                yc1 = m / np.power(p, 2) * xc1 * (2 * p - xc1)
                yc2 = m / np.power(1 - p, 2) * (1 - 2 * p + xc2) * (1 - xc2)
                # Y-coordinates of camber line
                yc = np.append(yc1, yc2)

                if self.cosine_spacing:
                    # points are generated according to cosine distribution of
                    # the X-coordinates of the chord
                    dyc1_dx = m / np.power(p, 2) * (2 * p - 2 * xc1)
                    dyc2_dx = m / np.power(1 - p, 2) * (2 * p - 2 * xc2)
                    dyc_dx = np.append(dyc1_dx, dyc2_dx)
                    theta = np.arctan(dyc_dx)
                    self.xup_coordinates = x - yt * np.sin(theta)
                    self.yup_coordinates = yc + yt * np.cos(theta)
                    self.xdown_coordinates = x + yt * np.sin(theta)
                    self.ydown_coordinates = yc - yt * np.cos(theta)
                else:
                    # Linear spacing distribution of the foil coordinates
                    self.xup_coordinates = np.linspace(
                        0.0, 1.0, num=self.n_points)
                    self.xdown_coordinates = np.linspace(
                        0.0, 1.0, num=self.n_points)
                    self.yup_coordinates = yc + yt
                    self.ydown_coordinates = yc - yt

        elif len(self.digits) == 5:
            # Returns n+1 points in [0 1] for the given 5-digits NACA string
            cld = float(self.digits[0]) * 0.15
            p = 5.0 * float(self.digits[1]) / 100.0
            s = float(self.digits[2])
            t = float(self.digits[3:]) / 100.0

            # half-thickness distribution
            yt = 5 * t * (a0 * np.sqrt(x) + a1 * x + a2 * np.power(x, 2) +
                          a3 * np.power(x, 3) + a4 * np.power(x, 4))

            if s == 1:
                # Relfex camber
                P = np.array([0.1, 0.15, 0.2, 0.25])
                M = np.array([0.13, 0.2170, 0.318, 0.441])
                K = np.array([51.99, 15.793, 6.520, 3.191])
            elif s == 0:
                # Standard camber
                P = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
                M = np.array([0.0580, 0.1260, 0.2025, 0.2900, 0.3910])
                K = np.array([361.4, 51.64, 15.957, 6.643, 3.230])
            else:
                raise ValueError(
                    'For NACA "LPSTT" the value of "S" can be either 0 or 1.')

            if p == 0:
                # Symmetric foil
                self.xup_coordinates = np.linspace(0.0, 1.0, num=self.n_points)
                self.yup_coordinates = yt
                self.xdown_coordinates = np.linspace(
                    0.0, 1.0, num=self.n_points)
                self.ydown_coordinates = -yt
            else:
                # Cambered foil
                spl_m = splrep(P, M)
                spl_k = splrep(M, K)
                m = splev(p, spl_m)
                k1 = splev(m, spl_k)
                xc1 = np.asarray([xx for xx in x if xx <= m])
                xc2 = np.asarray([xx for xx in x if xx > m])
                yc1 = k1 / 6.0 * (np.power(xc1, 3) - 3 * m * np.power(xc1, 2) +
                                  np.power(m, 2) * (3 - m) * xc1)
                yc2 = k1 / 6.0 * np.power(m, 3) * (1 - xc2)
                yc = np.append(yc1, yc2)

                if self.cosine_spacing:
                    # points are generated according to cosine distribution of
                    # the X-coordinates of the chord
                    zc = cld / 0.3 * yc
                    dyc1_dx = 1.0 / 6.0 * k1 * (
                        3 * np.power(xc1, 2) - 6 * m * xc1 + np.power(m, 2) *
                        (3 - m))
                    dyc2_dx = np.tile(-1.0 / 6.0 * k1 * np.power(m, 3),
                                      len(xc2))
                    dyc_dx = np.append(dyc1_dx, dyc2_dx)
                    theta = np.arctan(dyc_dx)
                    self.xup_coordinates = x - yt * np.sin(theta)
                    self.yup_coordinates = zc + yt * np.cos(theta)
                    self.xdown_coordinates = x + yt * np.sin(theta)
                    self.ydown_coordinates = zc - yt * np.cos(theta)
                else:
                    # Linear spacing distribution of the foil coordinates
                    self.xup_coordinates = np.linspace(
                        0.0, 1.0, num=self.n_points)
                    self.xdown_coordinates = np.linspace(
                        0.0, 1.0, num=self.n_points)
                    self.yup_coordinates = yc + yt
                    self.ydown_coordinates = yc - yt

        else:
            raise Exception

